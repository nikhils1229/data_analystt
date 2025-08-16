import os
from openai import OpenAI

# Initialize client once
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set; OpenAI access not available")

client = OpenAI(api_key=OPENAI_API_KEY)

def chat(messages, model="gpt-4o-mini", max_tokens=512, temperature=0.0):
    """
    Simple wrapper around OpenAI chat completion using new OpenAI SDK.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content
