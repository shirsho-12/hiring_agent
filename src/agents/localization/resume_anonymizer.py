import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agents.base_agent import BaseAgent
from src.prompts.localization import RESUME_ANONYMIZER_PROMPT
from src.config.config import ANONYMIZER_MODEL, BASE_URL, API_KEY, TEMPERATURE


class AnonymizationAgent(BaseAgent):
    """
    An agent that anonymizes resumes by removing PII and standardizing the format.

    This agent handles:
    - Removing personal identifiers (names, emails, phone numbers, etc.)
    - Standardizing location information
    - Removing or generalizing company names
    - Cleaning up formatting and removing artifacts
    """

    def __init__(self):
        """
        Initialize the ResumeAnonymizer agent.

        Args:
            model_name: Optional model name to use for anonymization.
                       Defaults to ANONYMIZER_MODEL from config.
        """
        super().__init__()
        self.llm = ChatOpenAI(
            model=ANONYMIZER_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE,
        )
        self.prompt_template = PromptTemplate.from_template(RESUME_ANONYMIZER_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess the text before anonymization.

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

    def run(self, resume_text: str) -> Dict[str, Any]:
        """
        Anonymize the resume text.

        Args:
            resume_text: The resume text to anonymize

        Returns:
            The anonymized text
        """
        try:
            self.logger.info("Starting resume anonymization...")

            # Preprocess the text
            cleaned_text = self._preprocess_text(resume_text)

            # Prepare the input for the LLM
            input_data = {
                "resume_text": cleaned_text,
            }

            # Run the anonymization
            anonymized_text = self.chain.invoke(input_data)

            self.logger.info("Resume anonymization completed successfully.")

            return anonymized_text

        except Exception as e:
            self.logger.error(f"Error during resume anonymization: {str(e)}")
            raise RuntimeError(f"Failed to anonymize resume: {str(e)}")

    def batch_process(self, resumes: Dict[str, str]) -> Dict[str, str]:
        """
        Process multiple resumes in batch.

        Args:
            resumes: Dictionary of {resume_id: resume_text} pairs

        Returns:
            Dictionary of {resume_id: anonymization_result} pairs
        """
        results = {}
        for resume_id, text in resumes.items():
            try:
                results[resume_id] = self.run(text)
            except Exception as e:
                self.logger.error(f"Error processing resume {resume_id}: {str(e)}")
                results[resume_id] = {"error": str(e), "success": False}
        return results
