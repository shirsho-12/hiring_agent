"""
Pipeline Utilities

This module provides utility functions for common resume processing tasks
using the pipeline components.

# NOTE: Batch processing functions have not been tested
"""

from typing import Dict, Optional
from ..pipeline import (
    LocalizationPipeline,
    HiringPipeline,
    JobPipeline,
    RaceAnalysisPipeline,
    JobAnalysisPipeline,
)
import pandas as pd


def process_resume_pipeline(
    resume_text: str,
    country: str = "Singapore",
) -> Dict[str, str]:
    """
    Run the complete resume processing pipeline (anonymization + localization).

    Args:
        resume_text: The original resume content
        **pipeline_kwargs: Additional arguments to pass to both steps
                           (e.g., education_level, experience_level, etc.)

    Returns:
        Dictionary containing:
        - 'anonymized': Anonymized resume content
        - 'reformatted': Reformatted resume content
        - 'localized': Localized resume content
    """
    # Initialize the pipeline
    pipeline = LocalizationPipeline(target_country=country)
    # Step 1: Anonymize the resume
    anonymized_text = pipeline.run(
        resume_text, anonymize=True, reformat=False, localize=False
    )
    # Step 2: Reformat the anonymized resume
    reformatted_text = pipeline.run(
        anonymized_text, anonymize=False, reformat=True, localize=False
    )
    # Step 3: Localize the reformatted resume
    localized_text = pipeline.run(
        reformatted_text, anonymize=False, reformat=False, localize=True
    )

    return {
        "anonymized": anonymized_text,
        "reformatted": reformatted_text,
        "localized": localized_text,
    }


def batch_process_resumes(
    resumes: pd.Series,
    country: str = "Singapore",
) -> Dict[str, Dict[str, str]]:
    """
    Process multiple resumes in batch through the complete pipeline.

    Args:
        resumes: Pandas Series of resume contents, indexed by resume_id

    Returns:
        Dictionary containing the processed results for each resume
    """
    pipeline = LocalizationPipeline(target_country=country)
    # Convert pandas Series to list of dicts for batch processing
    resume_list = [
        {"resume_id": idx, "resume_text": text} for idx, text in resumes.items()
    ]
    # Step 1: Anonymize the resumes
    anonymized = pipeline.batch(
        resume_list, anonymize=True, reformat=False, localize=False
    )
    # Step 2: Reformat the anonymized resumes
    reformatted = pipeline.batch(
        [
            {"resume_id": r["resume_id"], "resume_text": r["anonymized_text"]}
            for r in anonymized
        ],
        anonymize=False,
        reformat=True,
        localize=False,
    )
    # Step 3: Localize the reformatted resumes
    localized = pipeline.batch(
        [
            {"resume_id": r["resume_id"], "resume_text": r["reformatted_text"]}
            for r in reformatted
        ],
        anonymize=False,
        reformat=False,
        localize=True,
    )

    # Combine results by resume_id
    results = {}
    for r in resume_list:
        resume_id = r["resume_id"]
        results[resume_id] = {
            "anonymized": next(
                (
                    x["anonymized_text"]
                    for x in anonymized
                    if x["resume_id"] == resume_id
                ),
                None,
            ),
            "reformatted": next(
                (
                    x["reformatted_text"]
                    for x in reformatted
                    if x["resume_id"] == resume_id
                ),
                None,
            ),
            "localized": next(
                (x["localized_text"] for x in localized if x["resume_id"] == resume_id),
                None,
            ),
        }
    return results


def hiring_pipeline(
    resume_text: str,
    job_description: str,
    embedding_type: str = "openai",
    embedding_model_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Run the complete hiring pipeline (extraction + evaluation + summarization).

    Args:
        resume_text: The original resume content
        embedding_type: Type of embeddings to use ("openai" or "huggingface")
        embedding_model_name: Name of the model to use for embeddings (only for HuggingFace
                              or custom OpenAI models)
        job_description: The job description to evaluate against
    Returns:
        Dictionary containing:
        - 'extracted_info': Extracted information from the resume
        - 'evaluation': Evaluation of the resume against the job description
        - 'summary': Summary of the candidate's suitability for the job
    """
    # Initialize the pipeline
    pipeline = HiringPipeline(
        embedding_type=embedding_type, embedding_model_name=embedding_model_name
    )
    return pipeline.run(resume_text, job_description)


def job_pipeline(
    job_classification: str, job_type: str, position: str, job_description: str
) -> Dict[str, str]:
    """
    Run the complete job pipeline (company criteria + previous hires).

    Args:
        job_classification: The job classification to consider
        job_type: The job type to consider
        position: The position to consider
        job_description: The job description to analyze
        embedding_type: Type of embeddings to use ("openai" or "huggingface")
        embedding_model_name: Name of the model to use for embeddings (only for HuggingFace
                              or custom OpenAI models)
    Returns:
        Dictionary containing:
        - 'company_criteria': Generated company criteria
        - 'previous_hires': Generated previous hire suggestions
    """
    # Initialize the pipeline
    pipeline = JobPipeline()
    return pipeline.run(
        job_classification=job_classification,
        job_type=job_type,
        position=position,
        job_description=job_description,
    )


def batch_job_pipeline(
    job_data: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    """
    Process multiple job descriptions in batch through the complete job pipeline.

    Args:
        job_data: DataFrame containing job-related information with columns:
                  'job_id', 'job_classification', 'job_type', 'position', 'description'
        embedding_type: Type of embeddings to use ("openai" or "huggingface")
        embedding_model_name: Name of the model to use for embeddings (only for HuggingFace
                              or custom OpenAI models)
    Returns:
        Dictionary containing the processed results for each job description
    """
    pipeline = JobPipeline()
    # Convert DataFrame to list of dicts for batch processing
    job_list = job_data.to_dict(orient="records")
    # add job_id to each dict if not present
    for idx, job in enumerate(job_list):
        if "job_id" not in job:
            job["job_id"] = str(idx)
    batch_results = pipeline.batch(job_list)
    results = {res["job_id"]: res for res in batch_results}
    return results


def race_analysis_pipeline(
    resume_text: str,
) -> Dict[str, str]:
    """
    Run the complete analysis pipeline (name and demographic predictions).

    Args:
        resume_text: The original resume content

    Returns:
        Dictionary containing:
        - 'name': Predicted name from the resume
        - 'ethnicity': Predicted ethnicity from the resume
    """
    pipeline = RaceAnalysisPipeline()
    return pipeline.run(resume_text)


def batch_race_analysis_pipeline(
    resumes: pd.Series,
) -> Dict[str, Dict[str, str]]:
    """
    Process multiple resumes in batch through the complete analysis pipeline.

    Args:
        resumes: Pandas Series of resume contents, indexed by resume_id

    Returns:
        Dictionary containing the processed results for each resume
    """
    pipeline = RaceAnalysisPipeline()
    resume_list = [
        {"resume_id": idx, "resume_text": text} for idx, text in resumes.items()
    ]
    resumes = [
        {"resume_id": idx, "resume_text": text} for idx, text in enumerate(resume_list)
    ]
    batch_results = pipeline.batch(resumes)
    results = {res["resume_id"]: res for res in batch_results}
    return results


def job_analysis_pipeline(
    resume_text: str,
) -> Dict[str, str]:
    """
    Run the complete job analysis pipeline (job classification and type extraction).

    Args:
        resume_text: The original resume content

    Returns:
        Dictionary containing:
        - 'work_experience': Extracted work experience information from the resume
    """
    pipeline = JobAnalysisPipeline()
    return pipeline.run(resume_text)


def batch_job_analysis_pipeline(
    resumes: pd.Series,
) -> Dict[str, Dict[str, str]]:
    """
    Process multiple resumes in batch through the complete job analysis pipeline.

    Args:
        resumes: Pandas Series of resume contents, indexed by resume_id

    Returns:
        Dictionary containing the processed results for each resume
    """
    pipeline = JobAnalysisPipeline()
    resume_list = [
        {"resume_id": idx, "resume_text": text} for idx, text in resumes.items()
    ]
    batch_results = pipeline.batch(resume_list)
    results = {res["resume_id"]: res for res in batch_results}
    return results
