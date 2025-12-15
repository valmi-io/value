"""Gemini auto-instrumentation example with Value SDK."""

import os
from dotenv import load_dotenv

load_dotenv()

from value import initialize_sync
from value.instrumentation import auto_instrument
from google import genai

# Get agent secret from environment variable
AGENT_SECRET = os.getenv("VALUE_AGENT_SECRET", "your-agent-secret")

client = initialize_sync(agent_secret=AGENT_SECRET)
auto_instrument(["gemini"])

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not set in environment")
    exit(1)

gemini_client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"
prompt = "Write a short, fun poem about tracing."

print(f"Making call to {model}...")

try:
    with client.action_context(user_id="user123", anonymous_id="anon456") as ctx:
        response = gemini_client.models.generate_content(model=model, contents=[prompt])

        ctx.send(
            action_name="process_gemini_response",
            **{
                "value.action.description": f"Received response from {model}",
                "model": model,
                "response_length": len(response.text),
                "prompt_type": "creative_writing",
            },
        )

        print(f"\n{response.text}")

except Exception as e:
    print(f"Error: {e}")
