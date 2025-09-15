import re
import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from src.agents.base_agent import BaseAgent
from src.prompts.job import COMPANY_CRITERIA_GENERATOR_PROMPT
from src.config.config import JOB_MODEL, BASE_URL, API_KEY, TEMPERATURE


class CompanyCriteriaGeneratorAgent(BaseAgent):
    """
    An agent that generates company criteria based on job descriptions.

    This agent handles:
    - Analyzing job descriptions to identify key company attributes
    - Generating a list of criteria that the company's HR should consider
    - Providing a one-line description of each criterion
    """

    def __init__(self):
        """
        Initialize the CompanyCriteriaGenerator agent.

        Args:
            model_name: Optional model name to use for generating company criteria.
                       Defaults to JOB_MODEL from config.
        """
        super().__init__()
        self.llm = ChatOpenAI(
            model=JOB_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE,
        )
        self.prompt_template = PromptTemplate.from_template(
            COMPANY_CRITERIA_GENERATOR_PROMPT
        )
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text before generating company criteria.

        Args:
            text: The input text to preprocess

        Returns:
            Preprocessed text with common artifacts removed
        """
        # Remove hyperlinks
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove common artifacts and normalize whitespace
        text = text.encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def run(
        self,
        job_classification: str,
        job_type: str,
        position: str,
        job_description: str,
    ) -> str:
        """
        Generate company criteria based on the provided job description.

        Args:
            job_classification: The job classification to consider
            job_type: The job type to consider
            position: The position to consider
            job_description: The job description text to analyze

        Returns:
            A string containing the generated company criteria
        """
        preprocessed_text = self._preprocess_text(job_description)
        response = self.chain.invoke(
            {
                "job_description": preprocessed_text,
                "job_classification": job_classification,
                "job_type": job_type,
                "position": position,
            }
        )
        return response

    def batch(self, jobs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process multiple job descriptions in batch to generate company criteria.

        Args:
            jobs: List of job descriptions to process

        Returns:
            Dictionary of {job_id: company_criteria} pairs
        """
        results = []
        try:
            batch_inputs = [
                {
                    "job_description": self._preprocess_text(
                        job.get("description", "")
                    ),
                    "job_classification": job.get("job_classification", ""),
                    "job_type": job.get("job_type", ""),
                    "position": job.get("position", ""),
                }
                for job in jobs
            ]
            batch_outputs = self.chain.batch(batch_inputs)
            for idx, job in enumerate(jobs):
                job_id = job.get("job_id")
                results.append(
                    {"job_id": job_id, "company_criteria": batch_outputs[idx]}
                )
        except Exception as e:
            self.logger.error(
                f"Error during batch company criteria generation: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to generate company criteria in batch: {str(e)}"
            )
        return results
