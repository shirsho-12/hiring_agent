from src.agents.base_agent import BaseAgent
from src.prompts.resume.resume_reformatter_prompt import RESUME_REFORMATTER_PROMPT

class ResumeReformatter(BaseAgent):
    """An agent that reformats a resume to ensure it is clean, professional, and consistently structured."""

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def run(self, anonymized_resume_text: str) -> str:
        """Reformats the resume to ensure a clean and professional layout."""
        self.logger.info("Running resume reformatter")

        prompt = RESUME_REFORMATTER_PROMPT.format(
            anonymized_resume_text=anonymized_resume_text
        )

        response = self.llm.completion(prompt)
        return response
