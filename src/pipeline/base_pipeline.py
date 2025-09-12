from abc import ABC, abstractmethod
from src.utils.logger import get_logger


class BasePipeline(ABC):
    """An abstract base class for all pipelines."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, *args, **kwargs):
        """The main entry point for the pipeline's execution."""
        pass
