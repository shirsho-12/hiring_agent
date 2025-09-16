NAME_PROMPT = """
Given the following anonymized resume from Singapore, generate an appropriate name of the person:

{resume_text}

ONLY OUTPUT THE NAME. NO EXPLANATIONS OR ADDITIONAL TEXT.
"""


ETHNICITY_PROMPT = """
Given the following name from Singapore, predict the person's ethnicity. Choose one from [Chinese, Malay, Indian, Eurasian].
Name: {name}

ONLY OUTPUT THE ETHNICITY. NO EXPLANATIONS OR ADDITIONAL TEXT.
"""

GENDER_PROMPT = """
Given the following name from Singapore, predict the person's gender. Choose one from [Male, Female, Other].
Name: {name}

ONLY OUTPUT THE GENDER. NO EXPLANATIONS OR ADDITIONAL TEXT.
"""

AGE_PROMPT = """
Given the following anonymized resume from Singapore, predict the person's age range. Choose one from [18-24, 25-34, 35-44, 45-54, 55-64, 65+].
Resume: {resume_text}

ONLY OUTPUT THE AGE RANGE. NO EXPLANATIONS OR ADDITIONAL TEXT.
"""
