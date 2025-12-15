# Value Seed Demo Data

This tool seeds comprehensive demo data for the Value platform, including Control Plane entities and ClickHouse trace data.

## Features

- **Control Plane Seeding**:

  - 2 Agent Types
  - 5 Accounts per Agent Type (10 total)
  - 1 Contact per Account (10 total, 1:1 with accounts)
  - 1 Agent Instance per Account (10 total)
  - 1-10 Anonymous IDs generated per Agent (for round-robin usage)
  - Products and Rate Plans with Charge Models
  - Subscriptions for each Account
  - Cost Allocation Rules
  - Tagged Actions and Outcomes configuration
  - Default Invoice Template

- **Persistent Agent Data**:

  - Agent instance IDs and anonymous IDs saved to `data/seeded_agents.json`
  - Reused across `seed-past` and `seed-live` for consistent data
  - Round-robin anonymous ID selection (1-3 IDs per action)

- **ClickHouse Seeding**:
  - Historical trace data for configurable months (default: 4 months)
  - Realistic weekday/weekend patterns
  - Business hours distribution
  - Pipeline traces matching `live_data.py` patterns:
    - Object Detection
    - Object Classification
    - OCR Processing
    - RAG Retrieval
    - LLM Calls (Gemini/Ollama)
    - Response Processing

## Installation

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
cd value-seed-demo-data

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

```

## Configuration

Create a `.env` file (copy from `.env.example`):

```bash
# Control Plane Configuration (for seed-past, seed-live, and status)
CONTROL_PLANE_URL=http://localhost:8200

# ClickHouse Configuration (for seed-past)
CLICKHOUSE_HOST=localhost
CLICKHOUSE_HTTP_PORT=8123
CLICKHOUSE_DB=default
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=

# Seeding Configuration
SEED_EVENTS_PER_DAY=75
SEED_MONTHS=4

# Value SDK Configuration (for seed-live)
VALUE_BACKEND_URL=http://localhost:8200
VALUE_OTEL_ENDPOINT=http://localhost:4317

# PostgreSQL Configuration (for direct DB updates)
POSTGRES_HOST="localhost"
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database
```

## Usage

### Seed Historical Data (seed-past)

Seeds 4 months of historical data up to now():

```bash
# Full seeding (Control Plane + ClickHouse)
python seed_data.py seed-past

# Custom duration and volume
python seed_data.py seed-past --months 6 --events-per-day 100

# Skip Control Plane (only seed ClickHouse)
python seed_data.py seed-past --skip-control-plane

# Keep existing ClickHouse data
python seed_data.py seed-past --no-clear
```

### Seed Live Data (seed-live)

Continuously generates real-time trace data using the **Value SDK** (like `live_data.py`).

Requires:

- `data/seeded_agents.json` (created by `seed-past`)
- `VALUE_BACKEND_URL` in `.env` (Control Plane URL)
- `VALUE_OTEL_ENDPOINT` in `.env` (OpenTelemetry collector endpoint)

```bash
# Start live simulation (5 second intervals)
python seed_data.py seed-live

# Custom interval and pipelines per iteration
python seed_data.py seed-live --interval 3 --pipelines 2
```

This mode:

1. Loads agent data from `data/seeded_agents.json`
2. Refreshes secrets for each agent via Control Plane API
3. Initializes Value SDK clients for each agent
4. Runs pipeline simulations with round-robin anonymous ID selection (1-3 IDs per action)

Traces flow through: Value SDK → OpenTelemetry Collector → ClickHouse

Press `Ctrl+C` to stop the live simulation.

### Check Status

```bash
python seed_data.py status
```
