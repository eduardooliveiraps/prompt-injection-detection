"""this is the main file for the LLMGuard module.
It will contain the main class that will be used to create the LLMGuard object.
"""

from litellm import completion as llm_completion
from litellm.types.utils import ModelResponse, Choices, Message

from typing import Callable, Any


def default_input_guard(_messages) -> bool:
    return True


def default_output_guard(_messages) -> bool:
    return True


def create_invalid_response(text: str) -> ModelResponse:
    return ModelResponse(
        id="",
        object="chat.completion",
        created=0,
        model="",
        choices=[
            Choices(
                index=0,
                finish_reason="length",
                message=Message(
                    content=text,
                    role="assistant",
                ),
            )
        ],
    )


class LLMGaurd:
    def __init__(
        self,
        input_guard: Callable[[Any], bool] = None,
        output_guard: Callable[[Any], bool] = None,
        invalid_input_response: ModelResponse = None,
        invalid_output_response: ModelResponse = None,
    ):
        self.input_guard = input_guard if input_guard else default_input_guard
        self.output_guard = output_guard if output_guard else default_output_guard
        self.invalid_input_response = invalid_input_response
        self.invalid_output_response = invalid_output_response

    def completions(self, **kwargs):
        """create a completion using the completions resource object."""
        messages = kwargs.get("messages")

        if not self.input_guard(messages):
            return self.invalid_input_response

        output_response = llm_completion(**kwargs)

        if not self.output_guard(output_response):
            return self.invalid_output_response

        return output_response
