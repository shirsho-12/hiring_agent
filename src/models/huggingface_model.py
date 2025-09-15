from langchain_core.runnables import Runnable
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from src.models.base_model import BaseModel


class HuggingFaceModel(BaseModel, Runnable):
    """
    Model wrapper for HuggingFace's Chat API, compatible with LangChain Runnable interface.
    """

    def __init__(
        self,
        model_name="meta-llama/Llama-3.1-8B",
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
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            ),
        )

        self.llm = ChatHuggingFace(
            llm=llm,
            model=model_name,
            temperature=self.temperature,
        )

    def invoke(self, input, *args, **kwargs):
        """
        LangChain Runnable interface: invoke method.
        Args:
            input: The input prompt string.
        Returns:
            The model's response as a string.
        """
        try:
            self.logger.info("Invoking HuggingFace model...")
            response = self.llm.invoke(input, *args, **kwargs)
            return response
        except Exception as e:
            self.logger.error(
                f"Error invoking HuggingFace model: {str(e)}", exc_info=True
            )
            raise RuntimeError("Failed to invoke HuggingFace model") from e

    def stream(self, input, *args, **kwargs):
        """
        LangChain Runnable interface: stream method.
        Args:
            input: The input prompt string.
        Yields:
            Streaming chunks of the model's response.
        """
        try:
            self.logger.info("Streaming HuggingFace model response...")
            for chunk in self.llm.stream(input, *args, **kwargs):
                yield chunk
        except Exception as e:
            self.logger.error(
                f"Error streaming HuggingFace model: {str(e)}", exc_info=True
            )
            raise RuntimeError("Failed to stream HuggingFace model") from e

    def __call__(self, input, *args, **kwargs):
        # Optional: makes the class itself callable
        return self.invoke(input, *args, **kwargs)
