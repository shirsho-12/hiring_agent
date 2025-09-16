"""
Resume Analysis Pipeline

This module provides a unified interface for analyzing resumes,
including name and demographic predictions.
"""

from typing import Dict, List
from src.pipeline.base_pipeline import BasePipeline
from ..agents.analysis import OneShotNameAgent, OneShotResumeAgent


class RaceAnalysisPipeline(BasePipeline):
    """
    A pipeline for analyzing resumes to extract names and demographic information.
    """

    def __init__(self):
        """
        Initialize the analysis pipeline with agents for name and demographic extraction.
        """
        super().__init__()
        self.name_agent = OneShotResumeAgent(mode="name")
        self.ethnicity_agent = OneShotNameAgent(mode="ethnicity")

    def run(self, resume_content: str) -> Dict[str, str]:
        """
        Analyze a resume to extract the name and ethnicity.
        """
        try:
            name = self.name_agent.run(resume_content)
            ethnicity = self.ethnicity_agent.run(name)
            return {"name": name, "ethnicity": ethnicity}
        except Exception as e:
            self.logger.log(f"Error processing resume: {e}")
            raise RuntimeError(f"Failed to analyze resume: {str(e)}")

    def batch(self, resumes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process multiple resumes in batch.
        """
        results = []
        try:
            names = self.name_agent.batch(resumes)
            names = [
                {"resume_id": item["resume_id"], "name": item["extracted"]}
                for item in names
            ]
            ethnicities = self.ethnicity_agent.batch(names)
            for ethnicity in ethnicities:
                result = {
                    "resume_id": ethnicity["resume_id"],
                    "name": ethnicity["name"],
                    "ethnicity": ethnicity["extracted"],
                }
                results.append(result)
        except Exception as e:
            self.logger.log(f"Error processing batch: {e}")
            raise RuntimeError(f"Failed to analyze batch: {str(e)}")
        return results


class JobAnalysisPipeline(BasePipeline):
    """
    A pipeline for analyzing job descriptions to extract relevant information.
    """

    def __init__(self):
        """
        Initialize the job analysis pipeline with agents for job classification and type extraction.
        """
        super().__init__()
        self.job_extraction_agent = OneShotResumeAgent(mode="work_experience")

    def run(
        self,
        resume_text: str,
    ) -> Dict[str, str]:
        """
        Analyze a job description to extract classification and type.
        """
        try:
            classification = self.job_extraction_agent.run(resume_text)
            return {"work_experience": classification}
        except Exception as e:
            self.logger.log(f"Error processing job description: {e}")
            raise RuntimeError(f"Failed to analyze job description: {str(e)}")

    def batch(self, resumes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process multiple job descriptions in batch.
        """
        results = []
        try:
            classifications = self.job_extraction_agent.batch(resumes)
            for idx, resume in enumerate(resumes):
                result = {
                    "resume_id": resume["resume_id"],
                    "work_experience": classifications[idx]["extracted"],
                }
                results.append(result)
        except Exception as e:
            self.logger.log(f"Error processing batch of jobs: {e}")
            raise RuntimeError(f"Failed to analyze batch of jobs: {str(e)}")
        return results
