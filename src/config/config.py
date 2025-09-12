import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
# You can swap out the model names here. It's also possible to use environment
# variables to set these values for more flexibility.
DEFAULT_MODEL = "gpt-4o-mini"
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", DEFAULT_MODEL)
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", DEFAULT_MODEL)
REFORMATTER_MODEL = os.environ.get("REFORMATTER_MODEL", DEFAULT_MODEL)
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", DEFAULT_MODEL)
ANONYMIZER_MODEL = os.environ.get("ANONYMIZER_MODEL", DEFAULT_MODEL)
LOCALIZATION_MODEL = os.environ.get("LOCALIZATION_MODEL", DEFAULT_MODEL)

TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", 0.0))
BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
