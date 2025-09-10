import os
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.vectorstores import FAISS
from src.utils.rag_loader import create_vector_store_from_sources
from src.agents.base_agent import BaseAgent
from src.prompts.resume_evaluator_prompt import EVALUATOR_PROMPT
from src.prompts.parser_prompt import EVALUATION_PARSER_PROMPT
from src.config.config import EVALUATOR_MODEL


class ResumeEvaluatorAgent(BaseAgent):
    """Agent responsible for evaluating a resume based on a job description."""

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=EVALUATOR_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0.0,
        )
        self.prompt_template = PromptTemplate.from_template(EVALUATOR_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()
        self.vector_store = create_vector_store_from_sources("data/rag_sources")

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
        """Evaluates the resume against the job description."""
        try:
            self.logger.info("Starting resume evaluation...")

            # 1. Retrieve relevant chunks
            try:
                retriever = self.vector_store.as_retriever()
                retrieved_chunks = retriever.get_relevant_documents(job_description)
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
                retrieved_chunks_text = "\n".join(
                    [f"- {doc.page_content[:200]}..." for doc in retrieved_chunks]
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
