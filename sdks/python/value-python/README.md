# Value Python SDK

[![PyPI version](https://badge.fury.io/py/value-python.svg)](https://badge.fury.io/py/value-python)
[![CI](https://github.com/valmi-io/value-python/actions/workflows/ci.yml/badge.svg)](https://github.com/valmi-io/value-python/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/pypi/pyversions/value-python.svg)](https://pypi.org/project/value-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python SDK to track AI agents with Value actions and auto-instrument LLM calls (Gemini, LangChain).

## Features

- **Value Actions**: Track agent behavior using `action_context` with `user_id` and `anonymous_id`, send custom actions via `ctx.send()`
- **Auto-Instrumentation**: Automatically capture LLM calls from Gemini and LangChain with zero code changes
- **OpenTelemetry-Based**: Built on OpenTelemetry for standardized, vendor-neutral observability

## Installation

### Basic Installation

Install the core SDK without auto-instrumentation dependencies:

```bash
pip install value-python
```

### With Google GenAI Auto-Instrumentation

Install with Google Generative AI (Gemini) auto-instrumentation support:

```bash
pip install value-python[genai]
```

### With LangChain Auto-Instrumentation

Install with LangChain auto-instrumentation support:

```bash
pip install value-python[langchain]
```

### With All Auto-Instrumentation Libraries

Install with all supported auto-instrumentation libraries:

```bash
pip install value-python[all]
```

### Multiple Extras

You can also install multiple extras:

```bash
pip install value-python[genai,langchain]
```

## Supported Platforms

- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Operating Systems**: Linux, macOS, Windows

## Quick Start

### Basic Usage

```python
import asyncio
from value import initialize_async

async def main():
    # agent_secret is required
    client = await initialize_async(agent_secret="your-agent-secret")

    async def process_data(data: str) -> str:
        print(f"Processing data: {data}")
        await asyncio.sleep(0.5)
        result = data.upper()

        with client.action_context(user_id="user123", anonymous_id="anon456") as ctx:
            ctx.send(
                action_name="transform_data",
                **{"value.action.description": f"Transformed data from {len(data)} to {len(result)} characters"}
            )
        return result

    result = await process_data("hello async world")
    print(f"Result: {result}")

asyncio.run(main())
```

### Synchronous Usage

```python
from value import initialize_sync

# agent_secret is required
client = initialize_sync(agent_secret="your-agent-secret")

with client.action_context(user_id="user123", anonymous_id="anon456") as ctx:
    # Your code here
    ctx.send(action_name="my_action", **{"custom.attribute": "value"})
```

### Auto-Instrumentation

Enable automatic tracing for supported AI libraries:

```python
from value import initialize_sync, auto_instrument

# Initialize the client with agent_secret
client = initialize_sync(agent_secret="your-agent-secret")

# Auto-instrument specific libraries
auto_instrument(["gemini", "langchain"])

# Or auto-instrument all available libraries
auto_instrument()
```

### Google GenAI Example

```python
from value import initialize_sync, auto_instrument
from google import genai

# Initialize Value client with agent_secret and auto-instrument
client = initialize_sync(agent_secret="your-agent-secret")
auto_instrument(["gemini"])

# Use Gemini as usual - traces are automatically captured
gemini_client = genai.Client(api_key="your-api-key")
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Write a poem about tracing"]
)

print(response.text)
```

## Configuration

The `agent_secret` is passed directly to `initialize_sync()` or `initialize_async()`. Additional configuration can be set using environment variables:

| Variable               | Description                                | Default                 |
| ---------------------- | ------------------------------------------ | ----------------------- |
| `VALUE_OTEL_ENDPOINT`  | OpenTelemetry collector endpoint           | `http://localhost:4317` |
| `VALUE_BACKEND_URL`    | Value Control Plane backend URL            | Required                |
| `VALUE_SERVICE_NAME`   | Service name for OpenTelemetry resource    | `value-control-agent`   |
| `VALUE_CONSOLE_EXPORT` | Enable console span exporter for debugging | `false`                 |

## Supported Auto-Instrumentation Libraries

| Library                       | Extra       | Instrumentor                                        |
| ----------------------------- | ----------- | --------------------------------------------------- |
| Google Generative AI (Gemini) | `genai`     | `opentelemetry-instrumentation-google-generativeai` |
| LangChain                     | `langchain` | `opentelemetry-instrumentation-langchain`           |

## API Reference

### Core Functions

- `initialize_sync(agent_secret)` - Initialize a synchronous Value client
- `initialize_async(agent_secret)` - Initialize an asynchronous Value client
- `auto_instrument(libraries=None)` - Enable auto-instrumentation for specified libraries
- `uninstrument(libraries=None)` - Disable auto-instrumentation
- `get_supported_libraries()` - Get list of supported library names
- `is_library_available(library)` - Check if a library's instrumentation is installed

### Client Methods

- `action_context(user_id=None, anonymous_id=None)` - Create a context for sending actions
- `ctx.send(action_name, **attributes)` - Send an action with custom attributes

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/valmi-io/value-python.git
cd value-python

# Install dependencies
poetry install

# Install with all extras for development
poetry install --extras all
```

### Running Tests

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=value --cov-report=html

# Run specific test file
poetry run pytest tests/test_client.py
```

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type check
poetry run mypy src/
```

## Publishing

The package is automatically published to PyPI when a new release is created on GitHub.

### Manual Publishing

```bash
# Build the package
poetry build

# Publish to TestPyPI (for testing)
poetry publish -r testpypi

# Publish to PyPI
poetry publish
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Links

- [Documentation](https://github.com/valmi-io/value-python#readme)
- [PyPI Package](https://pypi.org/project/value-python/)
- [Issue Tracker](https://github.com/valmi-io/value-python/issues)
