# In this file, you can define the prompt for the resume extractor agent.
# This makes it easy to swap out prompts without changing the agent's code.

RESUME_EXTRACTOR_PROMPT = """
Extract the following key details from the resume provided:

1.  **Position Applied For**: Identify the position name and its level (e.g., Junior, Mid-level, Senior, Leadership).
2.  **Self-Evaluation**: Summarize the candidate's self-evaluation or personal summary.
3.  **Skills & Specialties**: List the key skills and specialties mentioned.
4.  **Work Experience**: For each role, extract the company name, duration of employment, and key responsibilities.
5.  **Basic Information**: Extract the candidate's name, contact information (email, phone), and location.
6.  **Education Background**: List the educational institutions, degrees obtained, and graduation dates.

Please format the output as a JSON object with the following keys:
`position_applied_for`, `self_evaluation`, `skills_and_specialties`, `work_experience`, `basic_information`, `education_background`.

Resume:
--- --- ---
{resume_text}
--- --- ---
"""
