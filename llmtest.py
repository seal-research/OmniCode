import os
from litellm import completion


os.environ["OPENROUTER_API_KEY"] = "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" # Replace with your actual OpenRouter API Key

messages = [
    {"role": "user", "content": "Write a short, inspiring poem about nature."}
]


response = completion(
    model="openrouter/openai/gpt-5-mini",
    messages=messages,

)
print(response.choices[0].message.content)