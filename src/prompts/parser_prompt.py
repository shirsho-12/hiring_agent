"""Prompts for parsing and validating evaluation results."""

EVALUATION_PARSER_PROMPT = """You are an expert at parsing and validating structured data. 
Your task is to parse the following evaluation response and ensure it has the correct format and required fields.

Required fields:
- self_evaluation_score (float): Score from 0-10
- skills_score (float): Score from 0-10
- experience_score (float): Score from 0-10
- basic_info_score (float): Score from 0-10
- education_score (float): Score from 0-10

Input evaluation text:
{evaluation_text}

Please extract and return a valid JSON object with the required fields. 
If the input is already valid JSON with all required fields, return it as-is.
If any required fields are missing or invalid, provide reasonable defaults (0.0).

Output must be a valid JSON object with exactly these fields:
{{
  "self_evaluation_score": float,
  "skills_score": float,
  "experience_score": float,
  "basic_info_score": float,
  "education_score": float
}}"""
