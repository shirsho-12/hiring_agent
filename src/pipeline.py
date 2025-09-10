import wandb
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.agents.resume_extractor import ResumeExtractorAgent
from src.agents.resume_evaluator import ResumeEvaluatorAgent
from src.agents.resume_summarizer import ResumeSummarizerAgent
from src.score_formatter import format_scores

class HiringPipeline:
    """Orchestrates the entire hiring agent pipeline."""

    def __init__(self, wandb_project: str = "hiring-agent-pipeline"):
        load_dotenv()
        self.logger = get_logger(self.__class__.__name__)
        self.extractor = ResumeExtractorAgent()
        self.evaluator = ResumeEvaluatorAgent()
        self.summarizer = ResumeSummarizerAgent()
        wandb.init(project=wandb_project)

    def run(self, resume_path: str, job_description: str):
        """Runs the full pipeline from resume extraction to final summary."""
        self.logger.info("--- Starting Hiring Pipeline ---")

        # Read resume text
        try:
            with open(resume_path, 'r') as file:
                resume_text = file.read()
        except FileNotFoundError:
            self.logger.error(f"Resume file not found at {resume_path}")
            return

        # 1. Resume Extractor
        extracted_details = self.extractor.run(resume_text)
        if not extracted_details:
            self.logger.error("Failed to extract details from resume. Aborting pipeline.")
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
        self.logger.info("Formatting scores...")
        formatted_scores = format_scores(evaluation_scores_json)
        total_score = sum(formatted_scores)
        self.logger.info(f"Formatted Scores: {formatted_scores}")
        self.logger.info(f"Total Score: {total_score} / 10")
        wandb.log({"formatted_scores": formatted_scores, "total_score": total_score})

        # 4. Resume Summarizer
        final_summary = self.summarizer.run(extracted_details, evaluation_scores_json)
        if not final_summary:
            self.logger.error("Failed to generate final summary.")
            return
        self.logger.info(f"\n--- Final Candidate Summary ---\n{final_summary}")
        wandb.log({"final_summary": final_summary})

        self.logger.info("--- Pipeline Finished ---")
        wandb.finish()
