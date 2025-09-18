"""
OpenRouter models using LiteLLM.
"""

import os
import sys
from typing import Literal

import litellm
from litellm.utils import Choices, Message, ModelResponse
from openai import BadRequestError

from app.log import log_and_print
from app.model import common
from app.model.common import ClaudeContentPolicyViolation, Model


class OpenRouterModel(Model):
    """
    Base class for creating Singleton instances of OpenRouter models.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        cost_per_input: float,
        cost_per_output: float,
        max_output_token: int = 4096,
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        self.max_output_token = max_output_token
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key.
        """
        self.check_api_key()

    def check_api_key(self) -> str:
        key_name = "OPENROUTER_API_KEY"
        key = os.getenv(key_name)
        if not key:
            print(f"Please set the {key_name} env var")
            sys.exit(1)
        return key

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    def call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ):
        # FIXME: ignore tools field since we don't use tools now
        if temperature is None:
            temperature = common.MODEL_TEMP

        try:

            if response_format == "json_object":
                last_content = messages[-1]["content"]
                last_content += "\nYour response should start with { and end with }. DO NOT write anything else other than the json."
                messages[-1]["content"] = last_content

            response = litellm.completion(
                model=self.name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_output_token,
                top_p=top_p,
                stream=False,
            )

            assert isinstance(response, ModelResponse)
            resp_usage = response.usage
            assert resp_usage is not None
            input_tokens = int(resp_usage.prompt_tokens)
            output_tokens = int(resp_usage.completion_tokens)
            cost = self.calc_cost(input_tokens, output_tokens)

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            first_resp_choice = response.choices[0]
            assert isinstance(first_resp_choice, Choices)
            resp_msg: Message = first_resp_choice.message
            content = self.extract_resp_content(resp_msg)

            return content, cost, input_tokens, output_tokens

        except litellm.exceptions.ContentPolicyViolationError:
            # claude sometimes send this error when writing patch
            log_and_print("Encountered claude content policy violation.")
            raise ClaudeContentPolicyViolation

        except BadRequestError as e:
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            raise e


class ClaudeSonnet4(OpenRouterModel):
    def __init__(self):
        super().__init__(
            "openrouter/anthropic/claude-sonnet-4", 0.000003, 0.000015, parallel_tool_call=True
        )
        self.note = "Latest Claude Sonnet 4.0 model via OpenRouter"
