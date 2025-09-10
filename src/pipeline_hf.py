import wandb
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.agents.resume_extractor import ResumeExtractorAgent
from src.agents.resume_evaluator_hf import ResumeEvaluatorAgentHF
from src.agents.resume_summarizer import ResumeSummarizerAgent


class HiringPipelineHF:
    """Orchestrates the hiring agent pipeline with HuggingFace embeddings."""

    def __init__(self, wandb_project: str = "hiring-agent-hf-pipeline"):
        """
        Initialize the pipeline with HuggingFace-based components.
        
        Args:
            wandb_project: Name of the Weights & Biases project for tracking
        """
        load_dotenv()
        self.logger = get_logger(self.__class__.__name__)
        self.extractor = ResumeExtractorAgent()
        self.evaluator = ResumeEvaluatorAgentHF()  # Using HF-based evaluator
        self.summarizer = ResumeSummarizerAgent()
        
        # Initialize Weights & Biases
        wandb.init(project=wandb_project)

    def run(self, resume_path: str, job_description: str):
        """
        Run the full pipeline with HuggingFace embeddings.
        
        Args:
            resume_path: Path to the resume file
            job_description: Job description to evaluate against
        """
        self.logger.info("--- Starting Hiring Pipeline with HuggingFace ---")

        # Read resume text
        try:
            with open(resume_path, "r") as file:
                resume_text = file.read()
        except FileNotFoundError:
            self.logger.error(f"Resume file not found at {resume_path}")
            return

        # Step 1: Extract details from resume
        self.logger.info("Extracting details from resume...")
        try:
            resume_details = self.extractor.run(resume_text)
            self.logger.info("Extracted Details:")
            self.logger.info(resume_details)
            
            # Log to W&B
            wandb.log({"extracted_details": resume_details})
        except Exception as e:
            self.logger.error(f"Error in resume extraction: {e}")
            return

        # Step 2: Evaluate resume against job description
        self.logger.info("Evaluating resume...")
        try:
            evaluation_scores = self.evaluator.run(resume_details, job_description)
            self.logger.info("Evaluation Scores (JSON):")
            self.logger.info(evaluation_scores)
            
            # Log to W&B
            wandb.log({"evaluation_scores_json": evaluation_scores})
        except Exception as e:
            self.logger.error(f"Error in resume evaluation: {e}")
            return

        # Step 3: Generate final summary
        self.logger.info("Generating final summary...")
        try:
            final_summary = self.summarizer.run(resume_details, evaluation_scores)
            self.logger.info("\n--- Final Candidate Summary ---")
            self.logger.info(final_summary)
            
            # Log to W&B
            wandb.log({"final_summary": final_summary})
        except Exception as e:
            self.logger.error(f"Error in generating summary: {e}")
            return

        self.logger.info("--- Pipeline Finished ---")
        
        # Log completion to W&B
        wandb.finish()
        
        return {
            "extracted_details": resume_details,
            "evaluation_scores": evaluation_scores,
            "final_summary": final_summary
        }
