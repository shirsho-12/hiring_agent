"""
Pipeline Utilities

This module provides utility functions for common resume processing tasks
using the pipeline components.
"""

from typing import Dict, Optional
from ..pipeline import LocalizationPipeline, HiringPipeline


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
        - 'anonymized': anonymized_text
        - 'reformatted': reformatted_text
        - 'localized': localized_text
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
    resumes: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """
    Process multiple resumes in batch through the complete pipeline.

    Args:
        resumes: Dictionary of {resume_id: resume_content} pairs

    Returns:
        Dictionary containing the processed results for each resume
    """
    results = {}
    for resume_id, resume_content in resumes.items():
        results[resume_id] = process_resume_pipeline(resume_content)
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
