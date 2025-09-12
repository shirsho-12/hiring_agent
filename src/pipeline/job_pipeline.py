"""
Job Detail Generation Pipeline

This module provides a unified interface for generating detailed job-related information,
including company criteria and previous hire suggestions.
"""

from src.pipeline.base_pipeline import BasePipeline
from ..agents.job import PreviousHireGeneratorAgent, CompanyCriteriaGeneratorAgent


class JobPipeline(BasePipeline):
    """
    A pipeline for generating job-related details such as company criteria and previous hires.
    """

    def __init__(self):
        """
        Initialize the job detail generation pipeline.
        """
        super().__init__()
        self.company_criteria_generator = CompanyCriteriaGeneratorAgent()
        self.previous_hire_generator = PreviousHireGeneratorAgent()

    def run(self, job_description: str) -> dict:
        """
        Generate job-related details based on the provided job description.

        Args:
            job_description: The job description text to analyze

        Returns:
            A dictionary containing generated company criteria and previous hires
        """

        try:
            # Step 1: Generate company criteria
            company_criteria = self.company_criteria_generator.run(
                job_description=job_description
            )
            self.logger.info("Generated Company Criteria:\n %s", company_criteria)

            # Step 2: Generate previous hires based on job description and company criteria
            previous_hires = self.previous_hire_generator.run(
                job_description=job_description, company_criteria=company_criteria
            )
            self.logger.info("Generated Previous Hires:\n %s", previous_hires)

            return {
                "company_criteria": company_criteria,
                "previous_hires": previous_hires,
            }
        except Exception as e:
            self.logger.error("Error in JobPipeline: %s", str(e))
            return {
                "company_criteria": None,
                "previous_hires": None,
            }
