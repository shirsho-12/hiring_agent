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

    def run(
        self,
        job_classification: str,
        job_type: str,
        position: str,
        job_description: str,
    ) -> dict:
        """
        Generate job-related details based on the provided job description.

        Args:
            job_classification: The job classification to consider
            job_type: The job type to consider
            position: The position to consider
            job_description: The job description text to analyze

        Returns:
            A dictionary containing generated company criteria and previous hires
        """

        try:
            # Step 1: Generate company criteria
            company_criteria = self.company_criteria_generator.run(
                job_classification=job_classification,
                job_type=job_type,
                position=position,
                job_description=job_description,
            )
            self.logger.info("Generated Company Criteria:\n %s", company_criteria)

            # Step 2: Generate previous hires based on job description and company criteria
            previous_hires = self.previous_hire_generator.run(
                job_classification=job_classification,
                job_type=job_type,
                position=position,
                job_description=job_description,
                company_criteria=company_criteria,
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

    def batch(self, jobs: list[dict]) -> list[dict]:
        """
        Process multiple job descriptions in batch through the complete job pipeline.

        Args:
            jobs: List of dictionaries containing job-related information with keys:
                  'job_id', 'job_classification', 'job_type', 'position', 'description'
        Returns:
            List of dictionaries containing the processed results for each job description
        """
        results = []
        try:
            # Step 1: Generate company criteria for all jobs
            company_criteria_results = self.company_criteria_generator.batch(jobs)

            # Merge company criteria back into jobs for previous hire generation
            for job, criteria in zip(jobs, company_criteria_results):
                job["company_criteria"] = criteria["company_criteria"]

            # Step 2: Generate previous hires for all jobs
            previous_hire_results = self.previous_hire_generator.batch(jobs)

            # Combine results
            for criteria, hires in zip(company_criteria_results, previous_hire_results):
                job_id = criteria["job_id"]
                results.append(
                    {
                        "job_id": job_id,
                        "company_criteria": criteria["company_criteria"],
                        "previous_hires": hires["previous_hires"],
                    }
                )
        except Exception as e:
            self.logger.error(f"Error during batch job processing: {str(e)}")
        return results
