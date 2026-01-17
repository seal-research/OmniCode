"""
For models other than those from OpenAI, use LiteLLM if possible.
"""

import os
import sys
from typing import Literal

import litellm
from litellm.utils import Choices, Message, ModelResponse
from openai import BadRequestError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.log import log_and_print
from app.model import common
from app.model.common import Model


class GeminiModel(Model):
    """
    Base class for creating Singleton instances of Gemini models.
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
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(name, cost_per_input, cost_per_output, parallel_tool_call)
        self._initialized = True

    def setup(self) -> None:
        """
        Check API key for OpenRouter.
        """
        self.check_api_key()



    def check_api_key(self) -> str:
        key_name = "OPENROUTER_API_KEY"

        api_key = os.getenv(key_name)
        if not api_key:
            print(f"Please set the {key_name} env var")
            print("Get your API key from: https://openrouter.ai/keys")
            sys.exit(1)
        return api_key

    def extract_resp_content(self, chat_message: Message) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_message.content
        if content is None:
            return ""
        else:
            return content

    @retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
    def call(
        self,
        messages: list[dict],
        top_p=1,
        tools=None,
        response_format: Literal["text", "json_object"] = "text",
        **kwargs,
    ):
        # FIXME: ignore tools field since we don't use tools now
        try:
            # For Gemini 2.5 Flash models, ensure no messages have empty content
            if self.name in ["openrouter/google/gemini-2.5-flash", "openrouter/google/gemini-2.5-flash-lite"]:
                # Debug: log the incoming messages
                model_name = "Gemini 2.5 Flash Lite" if "lite" in self.name else "Gemini 2.5 Flash"
                log_and_print(f"{model_name}: Processing {len(messages)} messages")
                for i, msg in enumerate(messages):
                    log_and_print(f"  Message {i}: role={msg.get('role', 'NONE')}, content_type={type(msg.get('content', None))}, content_len={len(str(msg.get('content', '')))}")
                
                validated_messages = []
                for i, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        log_and_print(f"  Skipping non-dict message {i}: {type(msg)}")
                        continue
                    
                    role = msg.get('role')
                    content = msg.get('content')
                    
                    # More strict validation
                    if not role:
                        log_and_print(f"  Skipping message {i}: no role")
                        continue
                    
                    if content is None:
                        log_and_print(f"  Skipping message {i}: content is None")
                        continue
                    
                    content_str = str(content).strip()
                    if not content_str:
                        log_and_print(f"  Skipping message {i}: empty content after strip")
                        continue
                    
                    # Special case: accept "." as valid content for tool calls
                    if content_str == ".":
                        log_and_print(f"  Accepting message {i}: special '.' content for tool calls")
                    
                    # Valid message
                    validated_messages.append({
                        'role': role,
                        'content': content_str
                    })
                    log_and_print(f"  Kept message {i}: role={role}, content_len={len(content_str)}")
                
                messages = validated_messages
                log_and_print(f"{model_name}: After validation, {len(messages)} messages remain")
                
                # If we have no valid messages, raise an error immediately
                if not messages:
                    log_and_print(f"ERROR: No valid messages remain for {model_name}")
                    raise ValueError(f"No valid messages with content for {model_name}")
            
            prefill_content = "{"
            if response_format == "json_object":  # prefill
                # Gemini 2.5 Flash models don't allow messages to end with assistant role
                if self.name not in ["openrouter/google/gemini-2.5-flash", "openrouter/google/gemini-2.5-flash-lite"]:
                    messages.append({"role": "assistant", "content": prefill_content})

            # Prepare completion arguments for OpenRouter
            completion_kwargs = {
                "model": self.name,
                "messages": messages,
                "temperature": common.MODEL_TEMP,
                "max_tokens": 1024,
                "top_p": top_p,
                "stream": False,
            }
            
            response = litellm.completion(**completion_kwargs)
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
            
            # Debug logging for Gemini 2.5 Flash models
            if self.name in ["openrouter/google/gemini-2.5-flash", "openrouter/google/gemini-2.5-flash-lite"]:
                model_name = "Gemini 2.5 Flash Lite" if "lite" in self.name else "Gemini 2.5 Flash"
                log_and_print(f"{model_name} response: content_type={type(content)}, content_len={len(str(content))}")
                log_and_print(f"{model_name} response preview: {repr(str(content)[:100])}")
                
                # Handle empty responses from Gemini 2.5 Flash models
                if not content or str(content).strip() == "":
                    log_and_print(f"WARNING: {model_name} returned empty content")
                    content = "I apologize, but I cannot provide a response to this request."
            
            if response_format == "json_object":
                # For Gemini 2.5 Flash models, we didn't add the prefill, so just ensure it starts with {
                if self.name in ["openrouter/google/gemini-2.5-flash", "openrouter/google/gemini-2.5-flash-lite"]:
                    if not content.startswith(prefill_content):
                        content = prefill_content + content
                else:
                    # For other models, prepend the prefilled character if needed
                    if not content.startswith(prefill_content):
                        content = prefill_content + content

            return content, cost, input_tokens, output_tokens

        except BadRequestError as e:
            if e.code == "context_length_exceeded":
                log_and_print("Context length exceeded")
            elif self.name in ["openrouter/google/gemini-2.5-flash", "openrouter/google/gemini-2.5-flash-lite"]:
                # Simple error logging for Gemini 2.5 Flash models - no complex logic
                model_name = "Gemini 2.5 Flash Lite" if "lite" in self.name else "Gemini 2.5 Flash"
                log_and_print(f"{model_name} error: {str(e)}")
            raise e


class GeminiPro(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini-1.0-pro-002", 0.00000035, 0.00000105, parallel_tool_call=True
        )
        self.note = "Gemini 1.0 from Google"


class Gemini15Pro(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-1.5-pro",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 1.5 from Google"


class Gemini20Flash(GeminiModel):
    def __init__(self):
        super().__init__(
            "gemini/gemini-2.0-flash",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 2.0 Flash from Google"


class Gemini25Flash(GeminiModel):
    def __init__(self):
        super().__init__(
            "openrouter/google/gemini-2.5-flash",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 2.5 Flash from Google via OpenRouter"


class Gemini25FlashLite(GeminiModel):
    def __init__(self):
        super().__init__(
            "openrouter/google/gemini-2.5-flash-lite",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Gemini 2.5 Flash Lite from Google via OpenRouter"


class Llama4Scout(GeminiModel):
    def __init__(self):
        super().__init__(
            "openrouter/meta-llama/llama-4-scout",
            0.00000035,
            0.00000105,
            parallel_tool_call=True,
        )
        self.note = "Llama 4 Scout from Meta via OpenRouter"
