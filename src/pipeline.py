import wandb
from dotenv import load_dotenv
from typing import Literal, Optional, Dict, Any
from src.utils.logger import get_logger
from src.agents.resume import (
    ResumeExtractorAgent,
    ResumeEvaluatorAgent,
    ResumeSummarizerAgent,
    ResumeAnonymizer,
    ResumeReformatter,
)


class HiringPipeline:
    """Orchestrates the entire hiring agent pipeline.

    This pipeline supports both HuggingFace and OpenAI embeddings.
    """

    def __init__(
        self,
        wandb_project: str = "hiring-agent-pipeline",
        embedding_type: Literal["openai", "huggingface"] = "openai",
        model_name: Optional[str] = None,
        llm: Optional[Any] = None,
    ):
        """Initialize the hiring pipeline.

        Args:
            wandb_project: Name of the Weights & Biases project
            embedding_type: Type of embeddings to use ("openai" or "huggingface")
            model_name: Name of the model to use for embeddings (only for HuggingFace)
            llm: An optional LLM instance for agents that require it
        """
        load_dotenv()
        self.logger = get_logger(self.__class__.__name__)

        # Initialize all agents
        self.extractor = ResumeExtractorAgent()
        self.evaluator = ResumeEvaluatorAgent(
            embedding_type=embedding_type, model_name=model_name
        )
        self.summarizer = ResumeSummarizerAgent()

        # Initialize new resume processing agents if LLM is provided
        if llm is not None:
            self.anonymizer = ResumeAnonymizer(llm)
            self.reformatter = ResumeReformatter(llm)

        wandb.init(project=wandb_project)

    def anonymize_resume(self, resume_text: str, country: str = "United States") -> str:
        """Anonymizes the resume by removing PII and standardizing location/company info.

        Args:
            resume_text: The raw resume text to be anonymized
            country: The target country for location standardization

        Returns:
            str: The anonymized resume text
        """
        if not hasattr(self, "anonymizer"):
            raise ValueError(
                "LLM not provided during initialization. Anonymization requires an LLM instance."
            )

        self.logger.info("Anonymizing resume...")
        return self.anonymizer.run(resume_text, country)

    def reformat_resume(self, anonymized_resume: str) -> str:
        """Reformats the resume to ensure clean and consistent formatting.

        Args:
            anonymized_resume: The anonymized resume text to be reformatted

        Returns:
            str: The reformatted resume text
        """
        if not hasattr(self, "reformatter"):
            raise ValueError(
                "LLM not provided during initialization. Reformating requires an LLM instance."
            )

        self.logger.info("Reformatting resume...")
        return self.reformatter.run(anonymized_resume)

    def process_resume(
        self, resume_text: str, country: str = "United States"
    ) -> Dict[str, str]:
        """Processes a resume through the full anonymization and reformatting pipeline.

        Args:
            resume_text: The raw resume text to be processed
            country: The target country for location standardization

        Returns:
            Dict containing the anonymized and reformatted resume
        """
        self.logger.info("Starting resume processing...")

        # Anonymize the resume
        anonymized = self.anonymize_resume(resume_text, country)

        # Reformat the anonymized resume
        reformatted = self.reformatter.run(anonymized)

        return {"anonymized": anonymized, "reformatted": reformatted}

    def run(self, resume_path: str, job_description: str):
        """Runs the full pipeline from resume extraction to final summary."""
        self.logger.info("--- Starting Hiring Pipeline ---")

        # Read resume text
        try:
            with open(resume_path, "r") as file:
                resume_text = file.read()
        except FileNotFoundError:
            self.logger.error(f"Resume file not found at {resume_path}")
            return

        # 1. Resume Extractor
        extracted_details = self.extractor.run(resume_text)
        if not extracted_details:
            self.logger.error(
                "Failed to extract details from resume. Aborting pipeline."
            )
            return
        self.logger.info(f"Extracted Details:\n{extracted_details}")
        wandb.log({"extracted_details": extracted_details})

        # 2. Resume Evaluator
        evaluation_scores_json = self.evaluator.run(extracted_details, job_description)
        if not evaluation_scores_json:
            self.logger.error("Failed to evaluate resume. Aborting pipeline.")
            return
        self.logger.info(f"Evaluation Scores (JSON):\n{evaluation_scores_json}")
        wandb.log({"evaluation_scores_json": evaluation_scores_json})

        # 3. Score Formatter
        # self.logger.info("Formatting scores...")
        # formatted_scores = format_scores(evaluation_scores_json)
        # total_score = sum(formatted_scores)
        # self.logger.info(f"Formatted Scores: {formatted_scores}")
        # self.logger.info(f"Total Score: {total_score} / 10")
        # wandb.log({"formatted_scores": formatted_scores, "total_score": total_score})

        # 4. Resume Summarizer
        final_summary = self.summarizer.run(extracted_details, evaluation_scores_json)
        if not final_summary:
            self.logger.error("Failed to generate final summary.")
            return
        self.logger.info(f"\n--- Final Candidate Summary ---\n{final_summary}")
        wandb.log({"final_summary": final_summary})

        self.logger.info("--- Pipeline Finished ---")
        wandb.finish()
