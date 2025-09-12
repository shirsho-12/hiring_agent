"""
Resume Processing Pipeline

This module provides a unified interface for processing resumes,
including localization and anonymization.
"""

from src.pipeline.base_pipeline import BasePipeline
from ..agents.localization.resume_anonymizer import AnonymizationAgent
from ..agents.localization.resume_reformatter import ResumeReformatterAgent
from ..agents.localization.resume_localizer import LocalizationAgent


class LocalizationPipeline(BasePipeline):
    """
    A pipeline for processing resumes with localization and anonymization.
    """

    def __init__(self, target_country: str = "Singapore"):
        """
        Initialize the resume processing pipeline.

        """
        super().__init__()
        self.anonymizer = AnonymizationAgent()
        self.reformatter = ResumeReformatterAgent()
        self.localizer = LocalizationAgent()
        self.target_country = target_country

    def run(
        self,
        resume_content: str,
        anonymize: bool = True,
        reformat: bool = True,
        localize: bool = True,
    ) -> str:
        """
        Process a resume with optional anonymization and localization.

        Args:
            resume_content: The resume content to process
            target_country: Target country/region code (e.g., 'SG', 'US', 'JP')
            anonymize: Whether to anonymize the resume
            localize: Whether to localize the resume

        Returns:
            processed_content: The processed resume content (or None on failure)
        """

        current_content = resume_content

        try:
            # Step 1: Anonymization
            if anonymize:
                anonymized_content = self.anonymizer.run(resume_text=current_content)
                self.logger.info("Anonymized resume:\n %s", anonymized_content)
                current_content = anonymized_content
            # Step 2: Reformatting
            if reformat:
                reformatted_content = self.reformatter.run(
                    anonymized_resume_text=current_content
                )
                self.logger.info("Reformatted resume:\n %s", reformatted_content)
                current_content = reformatted_content
            # Step 3: Localization
            if localize:
                localized_content = self.localizer.run(
                    resume_text=current_content,
                    target_country=self.target_country,
                )
                self.logger.info("Localized resume:\n %s", localized_content)
                current_content = localized_content
            return current_content
        except Exception as e:
            self.logger.error(f"Error processing resume: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to process resume") from e
