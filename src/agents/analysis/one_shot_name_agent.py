from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable

from src.agents.base_agent import BaseAgent
from src.prompts import ETHNICITY_PROMPT, GENDER_PROMPT
from src.config.config import DEFAULT_MODEL
from src.models.get_model import get_model


class OneShotNameAgent(BaseAgent):
    """
    An agent that extracts either ethnicity or gender from a name using a one-shot prompt.

    The agent can be initialized to use either the ethnicity prompt or the gender prompt.
    """

    def __init__(self, mode: str = "ethnicity"):
        """
        Initialize the OneShotNameAgent.

        Args:
            mode: Either "ethnicity" or "gender" to select the prompt.
        """
        super().__init__()
        self.llm = get_model(
            DEFAULT_MODEL,
            temperature=0.9,  # Use a high temperature for creativity
        )
        if mode == "ethnicity":
            self.prompt_template = PromptTemplate.from_template(ETHNICITY_PROMPT)
        elif mode == "gender":
            self.prompt_template = PromptTemplate.from_template(GENDER_PROMPT)
        else:
            raise ValueError("mode must be either 'ethnicity' or 'gender'")
        self.chain: Runnable = self.prompt_template | self.llm | StrOutputParser()

    def run(self, name: str) -> str:
        """
        Extract information from the name.

        Args:
            name: The name to process

        Returns:
            The extracted information (ethnicity or gender)
        """
        try:
            self.logger.debug(
                f"Running extraction with prompt: {self.prompt_template.template[:30]}..."
            )
            input_data = {"name": name}
            result = self.chain.invoke(input_data)
            self.logger.info("Extraction completed.")
            return result
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise RuntimeError(f"Failed to extract from name: {str(e)}")

    def batch(self, names: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Process multiple names in batch.

        Args:
            names: List of dictionaries containing resume_id and name

        Returns:
            List of dictionaries containing extraction results
        """
        results = []
        try:
            self.logger.debug(f"Processing batch of {len(names)} names.")
            batch_inputs = [{"name": item["name"]} for item in names]
            batch_outputs = self.chain.batch(batch_inputs)
            for idx, item in enumerate(names):
                results.append(
                    {
                        "resume_id": item["resume_id"],
                        "name": item["name"],
                        "extracted": batch_outputs[idx],
                    }
                )
            self.logger.info("Batch extraction completed.")
        except Exception as e:
            self.logger.error(f"Batch extraction failed: {e}")
            for item in names:
                results.append(
                    {
                        "resume_id": item["resume_id"],
                        "error": str(e),
                    }
                )
        return results
