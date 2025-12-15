"""Configuration constants for seed data generation."""

import os
from dotenv import load_dotenv

load_dotenv()

CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "http://localhost:8200")
WORKSPACE_ID = os.getenv("WORKSPACE_ID", "")
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID", "")

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_HTTP_PORT = int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123"))
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "default")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")

SEED_EVENTS_PER_DAY = int(os.getenv("SEED_EVENTS_PER_DAY", "75"))
SEED_MONTHS = int(os.getenv("SEED_MONTHS", "4"))

AGENT_TYPES = [
    {
        "type": "invoice_processing",
        "name": "Invoice Processing Agent",
        "description": "Processes and summarizes invoices using OCR and LLM",
    },
    {
        "type": "document_analysis",
        "name": "Document Analysis Agent",
        "description": "Analyzes documents for classification and extraction",
    },
]

ACTION_NAMES = [
    "object_detection",
    "object_classification",
    "ocr_processing",
    "rag_retrieval",
    "process_gemini_response",
    "process_ollama_response",
]

TAGGED_ACTIONS = [
    "gemini_llm_gemini-2.5-flash",
]

TAGGED_OUTCOMES = {
    "trace_events": ["process_ollama_response"],
    "auto_tags": ["gemini_llm_gemini-2.5-flash"],
}

LLM_MODELS = ["gemini-2.5-flash", "llama3"]

INVOICE_TYPES = ["invoice", "receipt", "bill", "purchase_order"]
VENDORS = ["Amazon", "Google", "Microsoft", "Apple", "Uber", "Lyft", "Walmart", "Target"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD"]

RATE_PLAN_CONFIG = {
    "name": "Standard Plan",
    "billing_cycle_code": "monthly",
    "currency_code": "USD",
}

CHARGE_MODELS = [
    {
        "name": "Flat Fee",
        "description": "Simple flat fee pricing - fixed amount regardless of usage",
        "type": "flat_fee",
        "pricing_strategy": {},
    },
    {
        "name": "Per Unit",
        "description": "Price per unit of usage - linear pricing",
        "type": "per_unit",
        "pricing_strategy": {},
    },
    {
        "name": "Tiered",
        "description": "Tiered pricing - different rates at different usage levels",
        "type": "tiered",
        "pricing_strategy": {
            "tiers": [
                {"from": 0, "to": 100, "price": 0.10},
                {"from": 101, "to": 1000, "price": 0.05},
                {"from": 1001, "to": None, "price": 0.02},
            ],
        },
    },
    {
        "name": "Volume",
        "description": "Volume-based pricing - all units priced at tier rate",
        "type": "volume",
        "pricing_strategy": {
            "tiers": [
                {"from": 0, "to": 100, "price": 0.10},
                {"from": 101, "to": 1000, "price": 0.05},
                {"from": 1001, "to": None, "price": 0.02},
            ],
        },
    },
]

AVAILABLE_CHARGES = [
    {
        "name": "Platform Fee",
        "description": "Monthly platform access fee",
        "charge_type": "recurring",
        "charge_model_type": "flat-fee",
        "list_price": 99.00,
    },
    {
        "name": "Per Action",
        "description": "Charge per action performed",
        "charge_type": "usage",
        "charge_model_type": "per-unit",
        "list_price": 0.01,
        "uom": "@action",
    },
    {
        "name": "Per Outcome",
        "description": "Charge per outcome generated",
        "charge_type": "usage",
        "charge_model_type": "tiered",
        "list_price": 0.05,
        "uom": "@outcome",
    },
    {
        "name": "Setup Fee",
        "description": "One-time setup fee",
        "charge_type": "one-time",
        "charge_model_type": "flat-fee",
        "list_price": 500.00,
    },
    {
        "name": "Per User",
        "description": "Charge per active user/seat",
        "charge_type": "usage",
        "charge_model_type": "per-unit",
        "list_price": 10.00,
        "uom": "@seat",
    },
    {
        "name": "API Call Fee",
        "description": "Charge per API call",
        "charge_type": "usage",
        "charge_model_type": "volume",
        "list_price": 0.001,
        "uom": "@action",
    },
]

COST_ALLOCATION_RULES = [
    {"action_value": "object_detection", "action_type": "trace_event", "measure": "unit", "cost": 1.5},
    {"action_value": "object_classification", "action_type": "trace_event", "measure": "unit", "cost": 1.2},
    {"action_value": "ocr_processing", "action_type": "trace_event", "measure": "unit", "cost": 1.3},
    {"action_value": "rag_retrieval", "action_type": "trace_event", "measure": "unit", "cost": 1.1},
    {"action_value": "gemini_llm_gemini-2.5-flash", "action_type": "auto_tag", "measure": "unit", "cost": 2.0},
    {"action_value": "process_gemini_response", "action_type": "trace_event", "measure": "unit", "cost": 1.4},
    {"action_value": "process_ollama_response", "action_type": "trace_event", "measure": "unit", "cost": 1.2},
]
