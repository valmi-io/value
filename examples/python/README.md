# Python Examples

## Examples

| File                        | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `basic_sync.py`             | Synchronous SDK usage with action context                    |
| `basic_async.py`            | Async SDK usage with action context                          |
| `gemini_instrumentation.py` | Auto-instrument Google Gemini API                            |
| `pipeline_simulation.py`    | Simulates complete AI pipeline (Detection → OCR → RAG → LLM) |
| `otel_basics.py`            | Raw OpenTelemetry tracing without Value SDK                  |

## Setup

```bash
pip install value-sdk python-dotenv

# Optional for Gemini example
pip install google-genai
```

## Environment Variables

Create a `.env` file:

```bash
VALUE_SECRET=your-agent-secret
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
GOOGLE_API_KEY=your-gemini-key  # For gemini_instrumentation.py
```

## Run

```bash
python basic_sync.py
python basic_async.py
python pipeline_simulation.py  # Runs continuously
```
