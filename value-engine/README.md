# Value Engine

Distributed data processing engine for telemetry ingestion, transformation, and analytics.

## Overview

Value Engine handles the real-time processing of agent telemetry data:

- **Telemetry Ingestion** — Receive traces via OpenTelemetry Collector
- **Stream Processing** — Kafka-based data pipelines
- **Task Orchestration** — Celery workers for async processing
- **Analytics Queries** — ClickHouse integration for fast aggregations
- **Invoice Generation** — Automated billing calculations

## Services

| Service                 | Description                            | Port |
| ----------------------- | -------------------------------------- | ---- |
| `server`                | REST API for queries and configuration | 8000 |
| `orchestrator`          | Pipeline coordination and scheduling   | —    |
| `celery-worker`         | Data processing tasks                  | —    |
| `celery-invoice-worker` | Invoice generation tasks               | —    |
| `celery-beat`           | Scheduled task triggers                | —    |
| `celery-flower`         | Task monitoring UI                     | 5555 |

## Data Pipelines

```
OTEL Collector → Kafka → Celery Workers → ClickHouse
                                      ↓
                              Live Metrics Tables
```

**Pipeline Stages**:

1. `trace` — Raw telemetry ingestion
2. `scratch` — Data flattening and tagging
3. `post` — Aggregation and metrics computation

## Configuration

Key environment variables:

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# ClickHouse
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_HTTP_PORT=8123
CLICKHOUSE_DB=default

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Control Plane
CONTROL_PLANE_URL=http://localhost:8200
```

````

### Docker

```bash
docker compose up server orchestrator celery-worker
````

## Monitoring

- **Celery Flower**: http://localhost:5555
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

## License

Elastic License 2.0 (ELv2) — See [LICENSE](./LICENSE)
