# This file allows the 'prompts' directory to be treated as a Python package.

from .analysis_prompt import NAME_PROMPT, ETHNICITY_PROMPT, GENDER_PROMPT, AGE_PROMPT

__all__ = ["NAME_PROMPT", "ETHNICITY_PROMPT", "GENDER_PROMPT", "AGE_PROMPT"]
