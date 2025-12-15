# Value SDK Examples

Code examples demonstrating how to instrument AI agents with the Value SDK.

## Python Examples

| Example                                                         | Description                      |
| --------------------------------------------------------------- | -------------------------------- |
| [basic_sync.py](./python/basic_sync.py)                         | Synchronous SDK usage            |
| [basic_async.py](./python/basic_async.py)                       | Async SDK usage                  |
| [gemini_instrumentation.py](./python/gemini_instrumentation.py) | Auto-instrument Gemini API calls |
| [pipeline_simulation.py](./python/pipeline_simulation.py)       | Complete AI pipeline simulation  |
| [otel_basics.py](./python/otel_basics.py)                       | Raw OpenTelemetry tracing        |

## Prerequisites

1. **Install the Value SDK**:

   ```bash
   pip install value-sdk
   ```

2. **Set environment variables**:

   ```bash
   export VALUE_SECRET=your-agent-secret
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   ```

3. **Start the platform** (if running locally):
   ```bash
   make value-up
   ```

## Running Examples

```bash
cd examples/python

# Basic examples
python basic_sync.py
python basic_async.py

# Gemini auto-instrumentation (requires GOOGLE_API_KEY)
export GOOGLE_API_KEY=your-key
python gemini_instrumentation.py

# Full pipeline simulation (runs continuously)
python pipeline_simulation.py
```

## SDK Quick Reference

### Initialize Client

```python
# Sync
from value import initialize_sync
client = initialize_sync()

# Async
from value import initialize_async
client = await initialize_async()
```

### Send Actions

```python
with client.action_context(user_id="user123", anonymous_id="anon456") as ctx:
    ctx.send(
        action_name="my_action",
        **{
            "value.action.description": "Action description",
            "custom_attribute": "value",
        },
    )
```

### Auto-Instrument LLMs

```python
from value.instrumentation import auto_instrument

auto_instrument(["gemini"])  # Supported: gemini, openai, anthropic
```

## License

MIT License — See [LICENSE](./LICENSE)
