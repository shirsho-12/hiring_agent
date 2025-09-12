"""
Prompt template for resume localization.

This prompt is designed to guide the LLM in localizing resumes based on
specific parameters like target country, education level, and experience.
"""

LOCALIZATION_PROMPT = """
You are an expert in resume localization and Singapore's hiring practices. Your task is to adapt the provided resume for the target country/region while maintaining its professional integrity and effectiveness.

## Resume Localization Parameters:
- Target Country: {target_country}

## Resume to Localize:
{resume_text}

## Resume Localization Guidelines:

### 1. Personal Information
- Format contact information according to local standards
- Adjust education system references (e.g., GPA to local grading system)
- Adjust name order if needed (e.g., family name first in some Asian countries)
- Localize address format

### 2. Professional Summary/Objective
- Adapt to local expectations for self-presentation
- Adjust level of formality
- Include relevant local keywords for ATS systems
- Align with local job market expectations

### 3. Work Experience
- Localize job titles to match target country's conventions
- Adapt company names if they have local equivalents
- Convert employment dates to local format
- Adjust work experience descriptions to highlight locally valued skills
- Quantify achievements using local metrics/currencies

### 4. Education
- Convert degree names to local equivalents
- Adapt institution names if they have local recognition
- Adjust grading systems (e.g., GPA to local scale)
- Include relevant local certifications or qualifications

### 5. Skills Section
- Localize technical terminology
- Include region-specific skills if relevant
- Adapt language proficiency descriptions to local standards
- Highlight skills most valued in the target job market

### 6. Formatting & Structure
- Use local resume format (chronological, functional, or hybrid)
- Adjust section ordering to match local preferences
- Use appropriate date formats
- Use local conventions for file naming

### 7. Cultural Adaptations
- Adjust level of directness in language
- Consider local business etiquette
- Adapt to local norms for self-promotion
- Be aware of cultural sensitivities

## Output Instructions:
1. Provide the fully localized resume content
2. Maintain the original meaning and professional impact
3. Preserve technical accuracy
4. Use clear, professional language
5. Format according to local standards

## Localized Resume:
[Your localized resume content here]
"""

# ## Additional Context (if any):
# - Education Level: {education_level}
# - Experience Level: {experience_level}
# - Target Job Title: {target_job_title}
# - Target Industry: {target_industry}
