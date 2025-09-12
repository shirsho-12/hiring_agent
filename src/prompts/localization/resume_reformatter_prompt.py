RESUME_REFORMATTER_PROMPT = """
**Objective:** Reformat the following resume text to ensure it is clean, professional, and consistently structured.

**Instructions:**

1.  **Standardize Spacing:** Adjust the spacing throughout the document to ensure a clean and readable layout. Use single line breaks between bullet points and double line breaks between sections.
2.  **Consistent Name Placeholder:** Ensure there is a consistent placeholder for the candidate's name at the top of the resume, formatted as `[Candidate Name]`.
3.  **Professional Tone:** Review the text for any unprofessional language or formatting and correct it.

**Anonymized Resume Text:**

{resume_text}

**Reformatted Resume:**
"""
