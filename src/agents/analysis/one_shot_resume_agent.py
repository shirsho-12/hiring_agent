from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agents.base_agent import BaseAgent
from src.prompts import NAME_PROMPT, AGE_PROMPT, WORK_EXPERIENCE_PROMPT
from src.config.config import DEFAULT_MODEL
from src.models.get_model import get_model


class OneShotResumeAgent(BaseAgent):
    """
    An agent that extracts either name or age from resumes using a one-shot prompt.

    The agent can be initialized to use either the name prompt or the age prompt.
    """

    def __init__(self, mode: str = "name"):
        """
        Initialize the OneShotResumeAgent.

        Args:
            mode: Either "name", "age", or "work_experience" to select the prompt.
        """
        super().__init__()
        self.llm = get_model(DEFAULT_MODEL)
        if mode == "name":
            self.prompt_template = PromptTemplate.from_template(NAME_PROMPT)
        elif mode == "age":
            self.prompt_template = PromptTemplate.from_template(AGE_PROMPT)
        elif mode == "work_experience":
            self.prompt_template = PromptTemplate.from_template(WORK_EXPERIENCE_PROMPT)
        else:
            raise ValueError("mode must be either 'name', 'age', or 'work_experience'")
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(self, resume_text: str) -> str:
        """
        Extract information from the resume text.

        Args:
            resume_text: The resume text to process

        Returns:
            The extracted information (name or age)
        """
        try:
            self.logger.debug(
                f"Running extraction with prompt: {self.prompt_template.template[:30]}..."
            )
            input_data = {"resume_text": resume_text}
            result = self.chain.invoke(input_data)
            self.logger.info("Extraction completed.")
            return result
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise RuntimeError(f"Failed to extract from resume: {str(e)}")

    def batch(self, resumes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process multiple resumes in batch.

        Args:
            resumes: List of dictionaries containing resume_id and resume_text

        Returns:
            List of dictionaries containing extraction results
        """
        results = []
        try:
            self.logger.debug(f"Processing batch of {len(resumes)} resumes.")
            batch_inputs = [
                {"resume_text": resume["resume_text"]} for resume in resumes
            ]
            batch_outputs = self.chain.batch(batch_inputs)
            for idx, resume in enumerate(resumes):
                results.append(
                    {
                        "resume_id": resume["resume_id"],
                        "extracted": batch_outputs[idx],
                    }
                )
            self.logger.info("Batch extraction completed.")
        except Exception as e:
            self.logger.error(f"Batch extraction failed: {e}")
            for resume in resumes:
                results.append(
                    {
                        "resume_id": resume["resume_id"],
                        "error": str(e),
                    }
                )
        return results
