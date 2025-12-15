# Value UI

Modern web dashboard for AI agent observability and management.

## Overview

Value UI provides a comprehensive interface for:

- **Agent Dashboard** — Real-time monitoring of agent activity
- **Trace Explorer** — Detailed view of actions, outcomes, and LLM calls
- **Usage Analytics** — Token consumption, costs, and performance metrics
- **Account Management** — Customer accounts and subscriptions
- **Billing & Invoices** — Invoice generation and cost analysis

## Features

- 📊 Real-time metrics dashboards
- 🔍 Searchable trace explorer
- 📈 Usage and cost analytics
- 👥 Multi-tenant workspace support
- 🌙 Dark mode support
- 📱 Responsive design

## Configuration

Key environment variables:

```bash
# API Endpoints
NEXT_PUBLIC_CONTROL_PLANE_URL=http://localhost:8200
NEXT_PUBLIC_ENGINE_URL=http://localhost:8000
```

## Access

- **Dashboard**: http://localhost:3000
- **Agents**: http://localhost:3000/agents
- **Metering**: http://localhost:3000/metering
- **Billing**: http://localhost:3000/billing

## License

Elastic License 2.0 (ELv2) — See [LICENSE](./LICENSE)
