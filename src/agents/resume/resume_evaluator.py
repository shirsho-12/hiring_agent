import os
import json
from typing import Optional, Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from src.agents.base_agent import BaseAgent
from src.prompts.resume import RESUME_EVALUATOR_PROMPT
from src.prompts.parser_prompt import EVALUATION_PARSER_PROMPT
from src.config.config import EVALUATOR_MODEL

# Import the appropriate RAG loader based on the embedding type
try:
    from src.utils.rag_loader_hf import RAGLoader as HFRAGLoader
except ImportError:
    HFRAGLoader = None

try:
    from src.utils.rag_loader import create_vector_store_from_sources
except ImportError:
    create_vector_store_from_sources = None


class ResumeEvaluatorAgent(BaseAgent):
    """Agent responsible for evaluating a resume based on a job description.

    This agent supports both HuggingFace and OpenAI embeddings.
    """

    def __init__(
        self,
        embedding_type: Literal["openai", "huggingface"] = "openai",
        model_name: Optional[str] = None,
    ):
        """Initialize the ResumeEvaluatorAgent.

        Args:
            embedding_type: Type of embeddings to use ("openai" or "huggingface")
            model_name: Name of the model to use for embeddings (only for HuggingFace)
        """
        super().__init__()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=EVALUATOR_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0.0,
        )

        # Set up the evaluation chain
        self.prompt_template = PromptTemplate.from_template(RESUME_EVALUATOR_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

        # Initialize the appropriate vector store
        self.embedding_type = embedding_type.lower()
        self.model_name = model_name
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store based on the embedding type."""
        try:
            if self.embedding_type == "huggingface":
                if HFRAGLoader is None:
                    raise ImportError(
                        "HuggingFace RAG loader not available. "
                        "Make sure to install the required dependencies."
                    )
                self.logger.info(
                    f"Initializing HuggingFace embeddings with model: {self.model_name}"
                )
                rag_loader = HFRAGLoader(
                    model_name=self.model_name
                    or "sentence-transformers/all-MiniLM-L6-v2"
                )
                return rag_loader.get_vector_store("data/rag_sources")

            elif self.embedding_type == "openai":
                if create_vector_store_from_sources is None:
                    raise ImportError(
                        "OpenAI RAG loader not available. "
                        "Make sure to install the required dependencies."
                    )
                self.logger.info("Initializing OpenAI embeddings")
                return create_vector_store_from_sources("data/rag_sources")

            else:
                raise ValueError(f"Unsupported embedding type: {self.embedding_type}")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def _create_error_response(self, error_msg: str) -> str:
        """Helper method to create a standardized error response."""
        return json.dumps(
            {
                "self_evaluation_score": 0,
                "skills_score": 0,
                "experience_score": 0,
                "basic_info_score": 0,
                "education_score": 0,
                "error": error_msg,
            },
            indent=2,
        )

    def run(self, resume_details: str, job_description: str) -> str:
        """Evaluates the resume against the job description.

        Args:
            resume_details: Extracted details from the resume
            job_description: Job description to evaluate against

        Returns:
            str: JSON string containing evaluation scores
        """
        try:
            self.logger.info("Starting resume evaluation...")

            # 1. Retrieve relevant chunks
            try:
                # Handle different vector store interfaces
                if hasattr(self.vector_store, "as_retriever"):
                    retriever = self.vector_store.as_retriever()
                    retrieved_chunks = retriever.get_relevant_documents(job_description)
                elif hasattr(self.vector_store, "similarity_search"):
                    retrieved_chunks = self.vector_store.similarity_search(
                        job_description, k=4
                    )
                else:
                    raise ValueError("Unsupported vector store interface")

                self.logger.info(
                    f"Retrieved {len(retrieved_chunks)} relevant document(s) for RAG."
                )

                if not retrieved_chunks:
                    raise ValueError(
                        "No documents were retrieved. The vector store might be empty."
                    )

            except Exception as e:
                error_msg = f"Error in retrieval: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return self._create_error_response(error_msg)

            # 2. Format retrieved chunks for the prompt
            try:
                # Handle both Document objects and strings
                def format_chunk(chunk):
                    content = (
                        chunk.page_content
                        if hasattr(chunk, "page_content")
                        else str(chunk)
                    )
                    return f"- {content[:200]}..."

                retrieved_chunks_text = "\n".join(
                    format_chunk(chunk) for chunk in retrieved_chunks
                )
            except Exception as e:
                error_msg = f"Error formatting chunks: {str(e)}"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg)

            # 3. Get the initial evaluation
            try:
                evaluation = self.chain.invoke(
                    {
                        "job_description": job_description,
                        "retrieved_chunks": retrieved_chunks_text,
                        "resume_details": resume_details,
                    }
                )
                self.logger.info("Initial evaluation completed successfully.")
                self.logger.debug(f"Evaluation response: {evaluation[:200]}...")
            except Exception as e:
                error_msg = f"Error in evaluation: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return self._create_error_response(error_msg)

            # 4. Parse and validate the response using LLM
            try:
                self.logger.info("Parsing and validating evaluation with LLM...")

                # Create a parser chain
                parser_prompt = PromptTemplate.from_template(EVALUATION_PARSER_PROMPT)
                parser_chain = (
                    {"evaluation_text": lambda x: x}
                    | parser_prompt
                    | self.llm
                    | StrOutputParser()
                )

                # Parse the evaluation
                parsed_evaluation = parser_chain.invoke(evaluation)
                self.logger.info("LLM parsing completed successfully.")

                # Parse the JSON response
                try:
                    evaluation_json = json.loads(parsed_evaluation)
                    self.logger.info(
                        f"Successfully parsed evaluation: {json.dumps(evaluation_json, indent=2)}"
                    )
                    return json.dumps(evaluation_json, indent=2)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse LLM output as JSON: {str(e)}")
                    # Fall back to the original evaluation if parsing fails
                    self.logger.info("Falling back to original evaluation output")
                    return evaluation

            except Exception as e:
                error_msg = f"Error in LLM parsing: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                # Return the original evaluation as a fallback
                return evaluation

        except Exception as e:
            error_msg = f"Unexpected error in resume evaluation: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)
