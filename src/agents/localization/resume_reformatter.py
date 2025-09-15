from typing import Dict, List
from src.agents.base_agent import BaseAgent
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from src.prompts.localization.resume_reformatter_prompt import RESUME_REFORMATTER_PROMPT
from src.config.config import REFORMATTER_MODEL
from src.models.get_model import get_model


class ResumeReformatterAgent(BaseAgent):
    """An agent that reformats a resume to ensure it is clean, professional, and consistently structured."""

    def __init__(self):
        super().__init__()
        self.llm = get_model(REFORMATTER_MODEL)
        self.prompt_template = PromptTemplate.from_template(RESUME_REFORMATTER_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(self, anonymized_resume_text: str) -> str:
        """Reformats the resume to ensure a clean and professional layout."""
        try:
            reformatted_resume = self.chain.invoke(
                {"resume_text": anonymized_resume_text}
            )
            return reformatted_resume
        except Exception as e:
            self.logger.error(f"Error during resume reformatting: {e}")
            raise RuntimeError("Failed to reformat resume") from e

    def batch(self, resumes: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Process multiple resumes in batch.

        Args:
            resumes: A list of dictionaries where each dictionary contains a resume ID and its text.
        Returns:
            A dictionary where keys are resume IDs and values are reformatted resume texts.
        """
        results = []
        try:
            batch_inputs = [
                {"resume_text": resume["resume_text"]} for resume in resumes
            ]
            batch_outputs = self.chain.batch(batch_inputs)
            for idx, resume in enumerate(resumes):
                results.append(
                    {
                        "resume_id": resume["resume_id"],
                        "reformatted_text": batch_outputs[idx],
                    }
                )
        except Exception as e:
            self.logger.error(f"Error processing batch reformatting: {str(e)}")
            for resume in resumes:
                results.append(
                    {
                        "resume_id": resume["resume_id"],
                        "error": str(e),
                    }
                )
        return results
