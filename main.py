from src.pipeline import HiringPipeline

if __name__ == "__main__":
    # Define the path to the resume and the job description
    resume_path = "data/sample_resume.txt"
    job_description = """
    We are looking for a Senior Software Engineer with extensive experience in Python, 
    cloud technologies (AWS), and building scalable microservices. The ideal candidate 
    should have strong leadership skills and a proven track record of mentoring junior 
    engineers. Experience with Django, React, and CI/CD is a plus.
    """

    # Initialize and run the pipeline
    pipeline = HiringPipeline()
    pipeline.run(resume_path, job_description)
