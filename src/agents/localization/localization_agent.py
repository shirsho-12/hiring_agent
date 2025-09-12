import os
from typing import Dict, Any, Optional
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agents.base_agent import BaseAgent
from src.prompts.localization import LOCALIZATION_PROMPT
from src.config.config import LOCALIZATION_MODEL, BASE_URL, API_KEY, TEMPERATURE


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
        self.llm = ChatOpenAI(
            model=LOCALIZATION_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE,
        )
        self.prompt_template = PromptTemplate.from_template(LOCALIZATION_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(
        self,
        resume_content: str,
        target_country: str = "Singapore",
        education_level: Optional[str] = None,
        experience_level: Optional[str] = None,
        target_job_title: Optional[str] = None,
        target_industry: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Localize a resume for a specific country/region.

        Args:
            resume_content: The resume content to localize
            target_country: The target country/region for localization
            education_level: Candidate's highest education level
                            (e.g., "bachelor", "master", "phd")
            experience_level: Candidate's experience level
                            (e.g., "entry", "mid", "senior")
            target_job_title: The job title/position being targeted
            target_industry: The industry of the target position
            additional_context: Optional additional context for localization

        Returns:
            localized_resume: The localized resume content
        """
        try:
            self.logger.info(f"Starting resume localization for {target_country}...")

            # Prepare the input for the LLM
            input_data = {
                "resume_content": resume_content,
                "target_country": target_country,
                "education_level": education_level or "Not specified",
                "experience_level": experience_level or "Not specified",
                "target_job_title": target_job_title or "Not specified",
                "target_industry": target_industry or "Not specified",
            }

            # Run the localization
            start_time = datetime.utcnow()
            localized_content = self.chain.invoke(input_data)
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            self.logger.info(
                f"Content localization completed in {processing_time:.2f} seconds"
            )

            return {
                "localized_resume": localized_content,
                "metadata": {
                    "target_country": target_country,
                    "processing_time_seconds": processing_time,
                    "localization_timestamp": datetime.utcnow().isoformat(),
                    "localization_parameters": {
                        "education_level": education_level,
                        "experience_level": experience_level,
                        "target_job_title": target_job_title,
                        "target_industry": target_industry,
                    },
                },
            }

        except Exception as e:
            self.logger.error(
                f"Error during content localization: {str(e)}", exc_info=True
            )
            return {
                "error": str(e),
                "success": False,
            }

    def batch_localize_resumes(
        self, resumes: Dict[str, str], target_country: str, **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Localize multiple resumes in batch.

        Args:
            resumes: Dictionary of {resume_id: resume_content} pairs
            target_country: The target country/region for localization
            **kwargs: Additional parameters to pass to localize_resume

        Returns:
            Dictionary of {resume_id: localization_result} pairs
        """
        results = {}
        for resume_id, resume_content in resumes.items():
            try:
                results[resume_id] = self.run(
                    resume_content=resume_content,
                    target_country=target_country,
                    **kwargs,
                )
            except Exception as e:
                self.logger.error(f"Error processing resume {resume_id}: {str(e)}")
                results[resume_id] = {"error": str(e), "success": False}
        return results
