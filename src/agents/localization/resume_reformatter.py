from langchain_openai import ChatOpenAI
from src.agents.base_agent import BaseAgent
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from src.prompts.localization.resume_reformatter_prompt import RESUME_REFORMATTER_PROMPT
from src.config.config import REFORMATTER_MODEL, BASE_URL, API_KEY, TEMPERATURE


class ResumeReformatter(BaseAgent):
    """An agent that reformats a resume to ensure it is clean, professional, and consistently structured."""

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(
            model=REFORMATTER_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            temperature=TEMPERATURE,
        )
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
