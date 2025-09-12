"""
Prompt template for resume localization.

This prompt is designed to guide the LLM in localizing resumes based on
specific parameters like target country, education level, and experience.
"""

LOCALIZATION_PROMPT = """
You are an expert in formatting to a {target_country}'s hiring practices. Your task is to adapt the provided resume.
## Resume Localization Parameters:
- Target Country: {target_country}


## Guidelines:
- Replace the following placeholders with companies and departments present in the target country, use real-world examples as much as possible:
   - Company names: [COMPANY X], [TECH COMPANY], [FINANCIAL INSTITUTION], etc.
   - Department names: [DEPARTMENT] or [TEAM NAME]
   - Manager/Supervisor names: [MANAGER], [SUPERVISOR]
   - Project names: [PROJECT X], [INTERNAL TOOL], etc.
- Convert employment dates to local format
- Convert degree names to local equivalents
- Replace [UNIVERSITY], [COLLEGE], [INSTITUTION] with actual names
- Localize course names if necessary
- Adjust grading systems to local scales (e.g. 4.0 GPA to 5.0 scale)
- Localize technical terminology
- Include region-specific skills if relevant
- Adapt language proficiency descriptions to local standards
- Highlight skills most valued in the target job market
- Use local resume format (chronological, functional, or hybrid)
- Adjust section ordering to match local preferences
- Use appropriate date formats

## Output Instructions:
1. Provide the fully localized resume content
2. Maintain the original meaning and professional impact
3. Preserve technical accuracy
4. Format according to local standards


## Input Resume:
{resume_text}


You are to output only the modified resume content without any additional commentary or explanations.
The final output should not have any placeholders for job or education related information.
## Result Resume:
[Model output]
"""

# ## Additional Context (if any):
# - Education Level: {education_level}
# - Experience Level: {experience_level}
# - Target Job Title: {target_job_title}
# - Target Industry: {target_industry}
