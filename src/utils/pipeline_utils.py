"""
Pipeline Utilities

This module provides utility functions for common resume processing tasks
using the pipeline components.

# NOTE: Batch processing functions have not been tested
"""

from typing import Dict, Optional
from ..pipeline import LocalizationPipeline, HiringPipeline, JobPipeline
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


def batch_hiring_pipeline(
    resumes: Dict[str, str],
    job_description: str,
    embedding_type: str = "openai",
    embedding_model_name: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Process multiple resumes in batch through the complete hiring pipeline.

    Args:
        resumes: Dictionary of {resume_id: resume_content} pairs
        job_description: The job description to evaluate against
        embedding_type: Type of embeddings to use ("openai" or "huggingface")
        embedding_model_name: Name of the model to use for embeddings (only for HuggingFace
                              or custom OpenAI models)
    Returns:
        Dictionary containing the processed results for each resume
    """
    results = {}
    for resume_id, resume_content in resumes.items():
        results[resume_id] = hiring_pipeline(
            resume_text=resume_content,
            job_description=job_description,
            embedding_type=embedding_type,
            embedding_model_name=embedding_model_name,
        )
    return results


def job_pipeline(
    job_description: str,
    embedding_type: str = "openai",
    embedding_model_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Run the complete job pipeline (company criteria + previous hires).

    Args:
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
    pipeline = JobPipeline(
        embedding_type=embedding_type, embedding_model_name=embedding_model_name
    )
    return pipeline.run(job_description)


def batch_job_pipeline(
    job_descriptions: Dict[str, str],
    embedding_type: str = "openai",
    embedding_model_name: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Process multiple job descriptions in batch through the complete job pipeline.

    Args:
        job_descriptions: Dictionary of {job_id: job_description} pairs
        embedding_type: Type of embeddings to use ("openai" or "huggingface")
        embedding_model_name: Name of the model to use for embeddings (only for HuggingFace
                              or custom OpenAI models)
    Returns:
        Dictionary containing the processed results for each job description
    """
    results = {}
    for job_id, job_description in job_descriptions.items():
        results[job_id] = job_pipeline(
            job_description=job_description,
            embedding_type=embedding_type,
            embedding_model_name=embedding_model_name,
        )
    return results
