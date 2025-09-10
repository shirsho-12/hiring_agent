# In this file, you can define the prompts for the resume summarizer sub-agents.

CEO_PROMPT = """
As the CEO, evaluate the candidate's leadership potential, strategic thinking, and overall business acumen based on their resume.

**Candidate Details:**
{resume_details}

**Evaluation Scores:**
{evaluation_scores}

Provide your feedback in a concise, professional manner, focusing on their potential to contribute to the company's high-level goals.
"""

CTO_PROMPT = """
As the CTO, assess the candidate's technical expertise, skills, and experience. Pay close attention to their proficiency with relevant technologies and their problem-solving abilities.

**Candidate Details:**
{resume_details}

**Evaluation Scores:**
{evaluation_scores}

Provide a detailed analysis of their technical strengths and weaknesses.
"""

HR_PROMPT = """
As the HR Manager, evaluate the candidate's soft skills, cultural fit, and overall professionalism. Look for evidence of teamwork, communication skills, and alignment with company values.

**Candidate Details:**
{resume_details}

**Evaluation Scores:**
{evaluation_scores}

Provide feedback on their suitability for the company culture and their potential for growth.
"""

FINAL_SUMMARY_PROMPT = """
As a hiring coordinator, your task is to synthesize the feedback from the CEO, CTO, and HR manager into a single, personalized summary for the candidate.

**CEO Feedback:**
{ceo_feedback}

**CTO Feedback:**
{cto_feedback}

**HR Feedback:**
{hr_feedback}

Combine these perspectives to provide structured and constructive feedback. Highlight the candidate's strengths and areas for improvement.
"""
