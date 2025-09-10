import os
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from src.utils.rag_loader_hf import RAGLoader
from src.agents.base_agent import BaseAgent
from src.prompts.resume_evaluator_prompt import EVALUATOR_PROMPT
from src.prompts.parser_prompt import EVALUATION_PARSER_PROMPT
from src.config.config import EVALUATOR_MODEL


class ResumeEvaluatorAgentHF(BaseAgent):
    """Agent responsible for evaluating a resume based on a job description using HuggingFace embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the ResumeEvaluatorAgent with HuggingFace embeddings.

        Args:
            model_name: Name of the HuggingFace model to use for embeddings
        """
        super().__init__()
        self.llm = ChatOpenAI(
            model=EVALUATOR_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0.0,
        )
        self.prompt_template = PromptTemplate.from_template(EVALUATOR_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

        # Initialize RAG loader with HuggingFace embeddings
        try:
            self.logger.info(f"Initializing RAG loader with model: {model_name}")
            self.rag_loader = RAGLoader(model_name=model_name)
            self.logger.info("Loading vector store from data/rag_sources")
            self.vector_store = self.rag_loader.get_vector_store("data/rag_sources")

            if not self.vector_store:
                self.logger.error(
                    "Failed to load vector store - get_vector_store returned None"
                )
            else:
                self.logger.info("Vector store loaded successfully")
                if hasattr(self.vector_store, "index") and hasattr(
                    self.vector_store.index, "ntotal"
                ):
                    self.logger.info(
                        f"Vector store contains {self.vector_store.index.ntotal} vectors"
                    )
                else:
                    self.logger.warning(
                        "Could not determine number of vectors in the vector store"
                    )

        except Exception as e:
            self.logger.error(
                f"Error initializing vector store: {str(e)}", exc_info=True
            )
            self.vector_store = None

    def run(self, resume_details: str, job_description: str) -> str:
        """
        Evaluates the resume against the job description using RAG with HuggingFace embeddings.

        Args:
            resume_details: Extracted details from the resume
            job_description: Job description to evaluate against

        Returns:
            str: JSON string containing evaluation scores
        """
        try:
            self.logger.info(
                "Starting resume evaluation with HuggingFace embeddings..."
            )

            # Validate vector store
            if not self.vector_store:
                error_msg = "Failed to initialize vector store"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg)

            # 1. Retrieve relevant chunks
            self.logger.info("Retrieving relevant chunks...")
            try:
                # Debug: Check vector store contents
                if hasattr(self.vector_store, "index") and hasattr(
                    self.vector_store.index, "ntotal"
                ):
                    self.logger.info(
                        f"Vector store contains {self.vector_store.index.ntotal} vectors"
                    )

                # Create retriever with error handling
                try:
                    retriever = self.vector_store.as_retriever(
                        search_kwargs={"k": 4}  # Limit to top 4 results
                    )
                    self.logger.info("Retriever created successfully")
                except Exception as e:
                    raise Exception(f"Failed to create retriever: {str(e)}")

                # Perform retrieval with error handling
                try:
                    retrieved_chunks = retriever.invoke(job_description)
                    self.logger.info(
                        f"Retrieved {len(retrieved_chunks)} relevant document(s) for RAG."
                    )
                    if not retrieved_chunks:
                        raise ValueError(
                            "No documents were retrieved. The vector store might be empty."
                        )
                except Exception as e:
                    raise Exception(f"Failed to retrieve documents: {str(e)}")

            except Exception as e:
                error_msg = f"Error in retrieval: {str(e)}"
                self.logger.error(error_msg, exc_info=True)  # Include full traceback
                return self._create_error_response(error_msg)

            # 2. Format chunks for the prompt
            try:
                retrieved_chunks_text = "\n".join(
                    [f"- {chunk.page_content[:200]}..." for chunk in retrieved_chunks]
                )
            except Exception as e:
                error_msg = f"Error formatting chunks: {str(e)}"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg)

            # 3. Prepare input for the LLM
            try:
                input_data = {
                    "resume_details": resume_details,
                    "job_description": job_description,
                    "retrieved_chunks": retrieved_chunks_text,
                }
                self.logger.info("Input data prepared for LLM.")
            except Exception as e:
                error_msg = f"Error preparing input data: {str(e)}"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg)

            # 4. Invoke the LLM chain
            try:
                self.logger.info("Invoking LLM chain...")
                evaluation = self.chain.invoke(input_data)
                self.logger.info("LLM chain completed successfully.")

                # Log first 200 chars of the response for debugging
                self.logger.info(
                    f"LLM response (truncated): {str(evaluation)[:200]}..."
                )
            except Exception as e:
                error_msg = f"Error in LLM chain: {str(e)}"
                self.logger.error(error_msg)
                import traceback

                self.logger.error(traceback.format_exc())
                return self._create_error_response(error_msg)

            # 5. Parse and validate the response using LLM
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
            return self._create_error_response(
                f"Unexpected error in resume evaluation: {str(e)}"
            )

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
