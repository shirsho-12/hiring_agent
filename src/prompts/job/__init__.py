"""
Prompts for the job RAG pipeline.

This module contains prompts used by the job-related agents to analyze job descriptions,
generate previous hire suggestions, and create company criteria.
"""

from .previous_hire_generator_prompt import PREVIOUS_HIRE_GENERATOR_PROMPT
from .company_criteria_generator_prompt import COMPANY_CRITERIA_GENERATOR_PROMPT

__all__ = [
    "PREVIOUS_HIRE_GENERATOR_PROMPT",
    "COMPANY_CRITERIA_GENERATOR_PROMPT",
]
