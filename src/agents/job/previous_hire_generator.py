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

    def run(self, job_description: str, company_criteria: str) -> str:
        """
        Generate a list of previous hires based on the provided job description.

        Args:
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
            }
        )
        return result
