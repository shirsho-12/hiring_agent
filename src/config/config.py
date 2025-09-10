import os

# --- Model Configuration ---
# You can swap out the model names here. It's also possible to use environment
# variables to set these values for more flexibility.
DEFAULT_MODEL = "gpt-4o-mini"
EXTRACTOR_MODEL = os.environ.get("EXTRACTOR_MODEL", DEFAULT_MODEL)
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", DEFAULT_MODEL)
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", DEFAULT_MODEL)
