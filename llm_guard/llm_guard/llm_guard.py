"""this is the main file for the LLMGuard module.
It will contain the main class that will be used to create the LLMGuard object.
"""

from litellm.main import Completions
from litellm.types.utils import ModelResponse

from typing import Callable, Any


def default_input_guard(_messages) -> bool:
    return True


def default_output_guard(_messages) -> bool:
    return True


class LLMGaurd:
    def __init__(
        self,
        completions: Completions,
        input_guard: Callable[[Any], bool] = None,
        output_guard: Callable[[Any], bool] = None,
        invalid_input_response: ModelResponse = None,
        invalid_output_response: ModelResponse = None,
    ):
        self.completions = completions
        self.input_guard = input_guard if input_guard else default_input_guard
        self.output_guard = output_guard if output_guard else default_output_guard
        self.invalid_input_response = invalid_input_response
        self.invalid_output_response = invalid_output_response

    def create(self, **kwargs):
        """create a completion using the completions resource object."""
        messages = kwargs.get("messages")

        if not self.input_guard(messages):
            return self.invalid_input_response

        output_response = self.completions.create(**kwargs)

        if not self.output_guard(output_response):
            return self.invalid_output_response

        return output_response
