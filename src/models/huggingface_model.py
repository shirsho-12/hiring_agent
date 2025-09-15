from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from src.models.base_model import BaseModel


class HuggingFaceModel(BaseModel):
    """
    Model wrapper for HuggingFace's Chat API.
    """

    def __init__(
        self,
        model_name,
        api_url=None,
        api_key=None,
        temperature=None,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ):
        super().__init__(api_url=api_url, api_key=api_key, temperature=temperature)
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs=dict(
                # max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            ),
        )

        self.llm = ChatHuggingFace(
            llm=llm,
            model=model_name,
            temperature=self.temperature,
        )

    def invoke(self, prompt: str, *args, **kwargs):
        """
        Invoke the HuggingFace model with a prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Additional parameters for the model.

        Returns:
            The model's response as a string.
        """
        try:
            self.logger.info("Invoking HuggingFace model...")
            response = self.llm.invoke(prompt, *args, **kwargs)
            return response
        except Exception as e:
            self.logger.error(
                f"Error invoking HuggingFace model: {str(e)}", exc_info=True
            )
            raise RuntimeError("Failed to invoke HuggingFace model") from e
