"""
Prompt template for the CompanyCriteriaGenerator agent.

This prompt is designed to guide the LLM in generating relevant company criteria
based on the provided job description.
"""

COMPANY_CRITERIA_GENERATOR_PROMPT = """
You are an expert HR analyst. Your task is to carefully review the following job description and extract the key company criteria that HR should consider when evaluating candidates or shaping company policies.

Instructions:
1. Analyze the job description to identify important company attributes, values, and requirements.
2. Generate a list of criteria that reflect what the company is seeking or prioritizing (e.g., culture, skills, experience, work environment, diversity, benefits, etc.).
3. For each criterion, provide a concise one-line description explaining its relevance or importance.

Job classification: {job_classification}
Job type: {job_type}
Position: {position}

Job Description:
{job_description}

Output Format:
- List each criterion as a bullet point.
- Include a one-line description for each criterion.
- Focus on clarity and relevance for HR decision-making.
Example Output:
- Criterion 1: [Description of criterion 1]
- Criterion 2: [Description of criterion 2]
- Criterion 3: [Description of criterion 3]
"""
