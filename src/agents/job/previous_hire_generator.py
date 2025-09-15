from src.agents.base_agent import BaseAgent
from src.prompts.job import PREVIOUS_HIRE_GENERATOR_PROMPT
from src.config.config import JOB_MODEL, BASE_URL, API_KEY, TEMPERATURE
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
import re
import os


class PreviousHireGeneratorAgent(BaseAgent):
    """
    An agent that generates a list of previous hires based on job descriptions.

    This agent handles:
    - Analyzing job descriptions to identify key skills and qualifications
    - Generating a list of previous hires who have held similar positions
    - Providing brief summaries of each candidate's relevant experience
    """

    def __init__(self):
        """
        Initialize the PreviousHireGenerator agent.

        Args:
            model_name: Optional model name to use for generating previous hires.
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
            PREVIOUS_HIRE_GENERATOR_PROMPT
        )
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text before generating previous hires.

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
        company_criteria: str,
    ) -> str:
        """
        Generate a list of previous hires based on the provided job description.

        Args:
            job_classification: The job classification to consider
            job_type: The job type to consider
            position: The position to consider
            job_description: The job description to analyze
            company_criteria: The company criteria to consider

        Returns:
            A string containing the generated list of previous hires
        """
        preprocessed_description = self._preprocess_text(job_description)
        result = self.chain.invoke(
            {
                "job_description": preprocessed_description,
                "company_criteria": company_criteria,
                "job_classification": job_classification,
                "job_type": job_type,
                "position": position,
            }
        )
        return result

    def batch(self, jobs: list[dict]) -> list[dict]:
        """
        Process multiple job descriptions in batch to generate previous hires.

        Args:
            jobs: List of dictionaries containing job-related information with keys:
                  'job_classification', 'job_type', 'position', 'description', 'company_criteria'

        Returns:
            List of dictionaries containing job IDs and their corresponding previous hires
        """
        results = []
        try:
            batch_inputs = [
                {
                    "job_description": self._preprocess_text(
                        job.get("description", "")
                    ),
                    "company_criteria": job.get("company_criteria", ""),
                    "job_classification": job.get("job_classification", ""),
                    "job_type": job.get("job_type", ""),
                    "position": job.get("position", ""),
                }
                for job in jobs
            ]
            batch_outputs = self.chain.batch(batch_inputs)
            for idx, job in enumerate(jobs):
                job_id = job.get("job_id")
                results.append({"job_id": job_id, "previous_hires": batch_outputs[idx]})
        except Exception as e:
            self.logger.error(f"Error during batch previous hire generation: {str(e)}")
            raise RuntimeError(f"Failed to generate previous hires in batch: {str(e)}")
        return results
