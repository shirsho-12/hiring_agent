import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agents.base_agent import BaseAgent
from src.prompts.resume import RESUME_EXTRACTOR_PROMPT
from src.config.config import EXTRACTOR_MODEL


class ResumeExtractorAgent(BaseAgent):
    """Agent responsible for extracting key details from a resume."""

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=EXTRACTOR_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            temperature=0.0,
        )
        self.prompt_template = PromptTemplate.from_template(RESUME_EXTRACTOR_PROMPT)
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(self, resume_text: str) -> str:
        """Extracts information from the resume text."""
        try:
            self.logger.info("Starting resume extraction...")
            extracted_data = self.chain.invoke({"resume_text": resume_text})
            self.logger.info("Resume extraction successful.")
            return extracted_data
        except Exception as e:
            self.logger.error(f"An error occurred during resume extraction: {e}")
            return ""
