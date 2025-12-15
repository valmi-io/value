"""Basic async example demonstrating Value SDK usage."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from value import initialize_async

# Get agent secret from environment variable
AGENT_SECRET = os.getenv("VALUE_AGENT_SECRET", "your-agent-secret")


async def main():
    client = await initialize_async(agent_secret=AGENT_SECRET)

    async def process_data(data: str) -> str:
        await asyncio.sleep(0.5)
        result = data.upper()

        with client.action_context(user_id="user123", anonymous_id="anon456") as ctx:
            ctx.send(
                action_name="transform_data",
                **{
                    "value.action.description": f"Transformed {len(data)} characters",
                    "input_length": len(data),
                    "output_length": len(result),
                    "transformation_type": "uppercase",
                },
            )

        return result

    result = await process_data("hello async world")
    print(f"Result: {result}")
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
