from typing import Dict, Any, List

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agents.base_agent import BaseAgent
from src.prompts.localization import LOCALIZATION_PROMPT
from src.config.config import LOCALIZATION_MODEL
from src.models.get_model import get_model


class LocalizationAgent(BaseAgent):
    """
    Agent responsible for localizing resumes based on specified parameters.

    This agent specializes in adapting resumes to different countries/regions by:
    - Converting education credentials to local equivalents
    - Adjusting work experience to match local expectations
    - Localizing skills and certifications
    - Formatting according to regional resume standards
    - Adapting personal information and contact details
    """

    def __init__(self):
        """
        Initialize the LocalizationAgent.
        """
        super().__init__()
        self.llm = get_model(LOCALIZATION_MODEL)
        self.prompt_template = PromptTemplate.from_template(LOCALIZATION_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(self, resume_text: str, target_country: str = "Singapore") -> str:
        """
        Localize a resume for a specific country/region.

        Args:
            resume_content: The resume content to localize
            target_country: The target country/region for localization

        Returns:
            localized_resume: The localized resume content
        """
        try:
            self.logger.info(f"Starting resume localization for {target_country}...")

            # Prepare the input for the LLM
            input_data = {
                "resume_text": resume_text,
                "target_country": target_country,
            }

            # Run the localization
            localized_content = self.chain.invoke(input_data)

            self.logger.info(f"Content localization completed successfully.")

            return localized_content

        except Exception as e:
            self.logger.error(
                f"Error during content localization: {str(e)}", exc_info=True
            )
            raise RuntimeError("Failed to localize resume") from e

    def batch(
        self, resumes: List[Dict[str, str]], target_country: str, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Localize multiple resumes in batch.

        Args:
            resumes: A list of dictionaries where each dictionary contains a resume ID and its text.
            target_country: The target country/region for localization
            **kwargs: Additional parameters to pass to localize_resume

        Returns:
            Dictionary of {resume_id: localization_result} pairs
        """
        results = []
        try:
            batch_inputs = [
                {
                    "resume_text": resume["resume_text"],
                    "target_country": target_country,
                    **kwargs,
                }
                for resume in resumes
            ]
            batch_outputs = self.chain.batch(batch_inputs)
            for idx, resume in enumerate(resumes):
                results.append(
                    {
                        "resume_id": resume["resume_id"],
                        "localized_text": batch_outputs[idx],
                    }
                )
        except Exception as e:
            self.logger.error(f"Error processing batch localization: {str(e)}")
            for resume in resumes:
                results.append(
                    {
                        "resume_id": resume["resume_id"],
                        "error": str(e),
                    }
                )
        return results
