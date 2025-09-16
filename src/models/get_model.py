from src.config.config import DEFAULT_API, DEFAULT_MODEL, TEMPERATURE

from src.models import HuggingFaceModel, OpenAIModel


def get_model(model_name: str = None, temperature: float = TEMPERATURE):
    """
    Returns a chat model instance based on the config setting and model name.

    Args:
        model_name (str, optional): The name of the model to use. Defaults to config value.

    Returns:
        An instance of the selected chat model.
    """
    model_name = model_name or DEFAULT_MODEL

    if DEFAULT_API == "huggingface":
        return HuggingFaceModel(model_name=model_name, temperature=temperature)
    elif DEFAULT_API in ("openrouter", "openai"):
        return OpenAIModel(model_name=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported DEFAULT_API: {DEFAULT_API}")
