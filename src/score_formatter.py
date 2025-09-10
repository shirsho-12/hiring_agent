import json
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)

def format_scores(scores_json: str) -> List[float]:
    """Formats the JSON output from the evaluator into a list of floats."""
    try:
        scores_dict = json.loads(scores_json)
        # The order of scores is important for consistency
        score_keys = [
            'self_evaluation_score',
            'skills_score',
            'experience_score',
            'basic_info_score',
            'education_score'
        ]
        return [float(scores_dict.get(key, 0.0)) for key in score_keys]
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error formatting scores: {e}")
        # Return a default list of zeros if formatting fails
        return [0.0, 0.0, 0.0, 0.0, 0.0]
