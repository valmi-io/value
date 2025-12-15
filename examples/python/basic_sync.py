"""Basic synchronous example demonstrating Value SDK usage."""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from value import initialize_sync

# Get agent secret from environment variable
AGENT_SECRET = os.getenv("VALUE_AGENT_SECRET", "your-agent-secret")


def main():
    client = initialize_sync(agent_secret=AGENT_SECRET)

    def process_data(data: str) -> str:
        result = data.upper()

        with client.action_context(user_id="user123", anonymous_id="anon456") as ctx:
            ctx.send(
                action_name="transform_data",
                **{
                    "value.action.description": f"Transformed {len(data)} characters",
                    "original_text": data,
                    "transformed_text": result,
                    "method": "uppercase",
                },
            )

        return result

    result = process_data("hello world")
    print(f"Result: {result}")
    time.sleep(2)


if __name__ == "__main__":
    main()
