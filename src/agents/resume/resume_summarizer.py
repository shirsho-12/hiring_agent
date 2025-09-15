from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agents.base_agent import BaseAgent
from src.prompts.resume import (
    CEO_PROMPT,
    CTO_PROMPT,
    HR_PROMPT,
    FINAL_SUMMARY_PROMPT,
)
from src.config.config import SUMMARIZER_MODEL
from src.models.get_model import get_model


class ResumeSummarizerAgent(BaseAgent):
    """Agent responsible for generating a personalized resume summary."""

    def __init__(self):
        super().__init__()
        self.llm = get_model(SUMMARIZER_MODEL)

    def _create_sub_agent_chain(self, prompt: str) -> Runnable:
        """Creates a chain for a sub-agent with the given prompt."""
        return PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()

    def run(self, resume_details: str, evaluation_scores: str) -> str:
        """Generates a summary based on multi-agent feedback."""
        try:
            self.logger.info("Generating feedback from sub-agents...")
            ceo_chain = self._create_sub_agent_chain(CEO_PROMPT)
            cto_chain = self._create_sub_agent_chain(CTO_PROMPT)
            hr_chain = self._create_sub_agent_chain(HR_PROMPT)

            ceo_feedback = ceo_chain.invoke(
                {
                    "resume_details": resume_details,
                    "evaluation_scores": evaluation_scores,
                }
            )
            self.logger.info("Generated CEO feedback.")
            cto_feedback = cto_chain.invoke(
                {
                    "resume_details": resume_details,
                    "evaluation_scores": evaluation_scores,
                }
            )
            self.logger.info("Generated CTO feedback.")
            hr_feedback = hr_chain.invoke(
                {
                    "resume_details": resume_details,
                    "evaluation_scores": evaluation_scores,
                }
            )
            self.logger.info("Generated HR feedback.")

            self.logger.info("Synthesizing final summary...")
            final_summary_chain = self._create_sub_agent_chain(FINAL_SUMMARY_PROMPT)
            final_summary = final_summary_chain.invoke(
                {
                    "ceo_feedback": ceo_feedback,
                    "cto_feedback": cto_feedback,
                    "hr_feedback": hr_feedback,
                }
            )
            self.logger.info("Final summary generated successfully.")

            return final_summary
        except Exception as e:
            self.logger.error(f"An error occurred during resume summarization: {e}")
            return ""
