"""
Prompts for the localization pipeline.

This module contains prompts used by the localization agents to adapt content
to different regions, industries, and professional contexts.
"""

from .localization_prompt import LOCALIZATION_PROMPT
from .resume_anonymizer_prompt import RESUME_ANONYMIZER_PROMPT
from .resume_reformatter_prompt import RESUME_REFORMATTER_PROMPT

__all__ = [
    "LOCALIZATION_PROMPT",
    "RESUME_ANONYMIZER_PROMPT",
    "RESUME_REFORMATTER_PROMPT",
]
