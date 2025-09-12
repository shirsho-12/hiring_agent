from typing import Literal, Optional
from src.pipeline.base_pipeline import BasePipeline
from src.agents.resume import (
    ResumeExtractorAgent,
    ResumeEvaluatorAgent,
    ResumeSummarizerAgent,
)


class HiringPipeline(BasePipeline):
    """Orchestrates the entire hiring agent pipeline.

    This pipeline supports both HuggingFace and OpenAI embeddings.
    """

    def __init__(
        self,
        embedding_type: Literal["openai", "huggingface"] = "openai",
        embedding_model_name: Optional[str] = None,
    ):
        """Initialize the hiring pipeline.

        Args:
            wandb_project: Name of the Weights & Biases project
            embedding_type: Type of embeddings to use ("openai" or "huggingface")
            model_name: Name of the model to use for embeddings (only for HuggingFace)
            llm: An optional LLM instance for agents that require it
        """
        super().__init__()
        # Initialize all agents
        self.extractor = ResumeExtractorAgent()
        self.evaluator = ResumeEvaluatorAgent(
            embedding_type=embedding_type, embedding_model_name=embedding_model_name
        )
        self.summarizer = ResumeSummarizerAgent()

    def run(self, resume_text: str, job_description: str):
        """Runs the full pipeline from resume extraction to final summary."""
        self.logger.info("--- Starting Hiring Pipeline ---")

        # 1. Resume Extractor
        extracted_details = self.extractor.run(resume_text)
        if not extracted_details:
            self.logger.error(
                "Failed to extract details from resume. Aborting pipeline."
            )
            return
        self.logger.info(f"Extracted Details:\n{extracted_details}")

        # 2. Resume Evaluator
        evaluation_scores_json = self.evaluator.run(extracted_details, job_description)
        if not evaluation_scores_json:
            self.logger.error("Failed to evaluate resume. Aborting pipeline.")
            return
        self.logger.info(f"Evaluation Scores (JSON):\n{evaluation_scores_json}")

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
        return {
            "extracted_details": extracted_details,
            "evaluation_scores_json": evaluation_scores_json,
            "final_summary": final_summary,
        }
