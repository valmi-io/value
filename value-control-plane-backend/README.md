# Value Control Plane Backend

FastAPI-based REST API for managing workspaces, agents, accounts, and billing configuration.

## Overview

The Control Plane serves as the central management layer for the Value Platform, providing:

- **Workspace Management** — Multi-tenant workspace isolation
- **Agent Registry** — Register and configure AI agents
- **Account Management** — Customer account and subscription handling
- **Billing Configuration** — Rate plans, charge models, and invoice generation
- **API Key Management** — Secure agent authentication

## API Endpoints

| Category   | Endpoints            |
| ---------- | -------------------- |
| Workspaces | `/api/v1/workspaces` |
| Agents     | `/api/v1/agents`     |
| Accounts   | `/api/v1/accounts`   |
| Rate Plans | `/api/v1/rate-plans` |
| Invoices   | `/api/v1/invoices`   |

## Configuration

Key environment variables:

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=value_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Server
PORT=8200
NUM_WORKERS=1
```

## API Documentation

Once running, access interactive docs at:

- **Swagger UI**: http://localhost:8200/docs
- **ReDoc**: http://localhost:8200/redoc

## License

Elastic License 2.0 (ELv2) — See [LICENSE](./LICENSE)
