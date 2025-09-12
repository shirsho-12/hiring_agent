"""
Resume Processing Pipeline

This package provides tools for processing resumes, including localization and
anonymization.
"""

from .localization_pipeline import LocalizationPipeline
from .hiring_pipeline import HiringPipeline

__all__ = ["LocalizationPipeline", "HiringPipeline"]
