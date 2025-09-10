import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.vectorstores import FAISS
from src.utils.rag_loader import create_vector_store_from_sources
from src.agents.base_agent import BaseAgent
from src.prompts.resume_evaluator_prompt import EVALUATOR_PROMPT
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

    def run(self, resume_details: str, job_description: str) -> str:
        """Evaluates the resume against the job description."""
        try:
            self.logger.info("Starting resume evaluation...")
            retriever = self.vector_store.as_retriever()
            retrieved_chunks = retriever.get_relevant_documents(job_description)
            self.logger.info(
                f"Retrieved {len(retrieved_chunks)} relevant document(s) for RAG."
            )

            # Format retrieved chunks for the prompt
            retrieved_chunks_text = "\n".join(
                [doc.page_content for doc in retrieved_chunks]
            )

            evaluation = self.chain.invoke(
                {
                    "job_description": job_description,
                    "retrieved_chunks": retrieved_chunks_text,
                    "resume_details": resume_details,
                }
            )
            self.logger.info("Resume evaluation successful.")
            return evaluation
        except Exception as e:
            self.logger.error(f"An error occurred during resume evaluation: {e}")
            return ""
