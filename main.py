from src.utils.pipeline_utils import process_resume_pipeline, hiring_pipeline
import wandb
import time
from dotenv import load_dotenv
import os

# os.environ["WANDB_DISABLED"] = "true"

curr_time = time.strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    # Define the path to the resume and the job description
    resume_path = "data/sample_resume_strong_sde.txt"
    job_description = """
    We are looking for a Senior Software Engineer with extensive experience in Python, 
    cloud technologies (AWS), and building scalable microservices. The ideal candidate 
    should have strong leadership skills and a proven track record of mentoring junior 
    engineers. Experience with Django, React, and CI/CD is a plus.
    """

    load_dotenv()
    wandb.login()
    wandb.init(project="hiring-agent-pipeline", name=f"run-{curr_time}")

    with open(resume_path, "r") as file:
        resume_content = file.read()

    # Initialize and run the pipeline
    resume_dct = process_resume_pipeline(resume_content, country="Singapore")
    wandb.log(resume_dct)

    final_resume = hiring_pipeline(
        resume_text=resume_dct["localized"],
        job_description=job_description,
        embedding_type="huggingface",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    wandb.log(final_resume)

    wandb.finish()
