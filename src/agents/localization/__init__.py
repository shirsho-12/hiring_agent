"""
Localization agents for adapting content to different regions and contexts.

This module contains agents that handle the adaptation of content such as resumes,
job descriptions, and other professional documents to different locales, industries,
and experience levels.
"""

from .localization_agent import LocalizationAgent
from .resume_anonymizer import AnonymizationAgent

__all__ = [
    "LocalizationAgent",
    "AnonymizationAgent",
]
