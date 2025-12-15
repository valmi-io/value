"""Control Plane API seeder for creating entities."""

import random
import httpx
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import structlog
import psycopg2

from .config import (
    CONTROL_PLANE_URL,
    AGENT_TYPES,
    RATE_PLAN_CONFIG,
    AVAILABLE_CHARGES,
    COST_ALLOCATION_RULES,
    TAGGED_ACTIONS,
    TAGGED_OUTCOMES,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
)
from .seeded_data import SeededDataManager

logger = structlog.get_logger(__name__)


class ControlPlaneSeeder:
    def __init__(self, seed_months: int = 4):
        self.seed_months = seed_months
        self.base_url = CONTROL_PLANE_URL.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        self.client = httpx.Client(timeout=30.0)
        self._postgres_conn = None

        self.workspace_id: Optional[str] = None
        self.organization_id: Optional[str] = None
        self.agent_types: List[Dict] = []
        self.accounts: List[Dict] = []
        self.contacts: List[Dict] = []
        self.agent_instances: List[Dict] = []
        self.products: List[Dict] = []
        self.rate_plans: List[Dict] = []
        self.subscriptions: List[Dict] = []
        self.triggers: List[Dict] = []
        self.charge_models: List[Dict] = []

    def _get_postgres_connection(self):
        if self._postgres_conn is None or self._postgres_conn.closed:
            if not POSTGRES_DB:
                logger.warning("POSTGRES_DB not set, cannot update activated_at in database")
                return None
            try:
                self._postgres_conn = psycopg2.connect(
                    host=POSTGRES_HOST,
                    port=POSTGRES_PORT,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    database=POSTGRES_DB,
                )
            except Exception as e:
                logger.error("Failed to connect to Postgres", error=str(e))
                return None
        return self._postgres_conn

    def _update_subscription_activated_at(self, subscription_id: str, activated_at: datetime) -> bool:
        conn = self._get_postgres_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE subscriptions SET activated_at = %s WHERE id = %s",
                    (activated_at, subscription_id),
                )
                conn.commit()
                logger.info(
                    "Updated subscription activated_at",
                    subscription_id=subscription_id,
                    activated_at=activated_at.isoformat(),
                )
                return True
        except Exception as e:
            logger.error("Failed to update subscription activated_at", subscription_id=subscription_id, error=str(e))
            conn.rollback()
            return False

    def _make_request(
        self, method: str, endpoint: str, json_data: Optional[Dict] = None, params: Optional[Dict] = None
    ) -> Dict:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.client.request(method=method, url=url, headers=self.headers, json=json_data, params=params)
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            logger.error(
                "API request failed",
                method=method,
                endpoint=endpoint,
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            raise
        except Exception as e:
            logger.error("API request error", method=method, endpoint=endpoint, error=str(e))
            raise

    def get_current_workspace(self) -> Dict:
        response = self._make_request("GET", "/api/v1/auth/me")
        self.workspace_id = response.get("workspace_id")
        self.organization_id = response.get("organization_id")
        logger.info(
            "Got workspace info from API",
            workspace_id=self.workspace_id,
            organization_id=self.organization_id,
            workspace_name=response.get("workspace_name"),
        )
        return response

    def get_triggers(self) -> List[Dict]:
        response = self._make_request("GET", "/api/v1/subscription/triggers")
        self.triggers = response.get("triggers", [])
        logger.info("Fetched triggers", count=len(self.triggers))
        return self.triggers

    def create_agent_types(self) -> List[Dict]:
        created = []
        for agent_type_config in AGENT_TYPES:
            try:
                response = self._make_request(
                    "POST",
                    "/api/v1/agent_type/create",
                    json_data={
                        "type": agent_type_config["type"],
                        "name": agent_type_config["name"],
                        "description": agent_type_config["description"],
                    },
                )
                created.append(response)
                logger.info("Created agent type", name=agent_type_config["name"], id=response.get("agent_type_id"))
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    logger.warning("Agent type may already exist", name=agent_type_config["name"])
                    existing = self._make_request("GET", "/api/v1/agent_type/all")
                    for at in existing.get("agent_types", []):
                        if at.get("name") == agent_type_config["name"]:
                            created.append(at)
                            logger.info("Found existing agent type", name=at.get("name"), id=at.get("agent_type_id"))
                            break
                else:
                    raise
        self.agent_types = created
        return created

    def create_accounts(self, count_per_agent_type: int = 5) -> List[Dict]:
        created = []
        for agent_type in self.agent_types:
            agent_type_id = agent_type.get("agent_type_id")
            agent_type_name = agent_type.get("name", "Unknown")
            for i in range(count_per_agent_type):
                try:
                    account_name = f"{agent_type_name} - Account {i + 1}"
                    response = self._make_request(
                        "POST",
                        "/api/v1/account/create",
                        json_data={
                            "fields": {
                                "name": account_name,
                                "description": f"Account for {agent_type_name}",
                                "external_id": f"acc_{agent_type_id}_{i + 1}",
                            }
                        },
                    )
                    response["_agent_type_id"] = agent_type_id
                    created.append(response)
                    logger.info("Created account", name=account_name, id=response.get("account_id"))
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 409:
                        logger.warning("Account may already exist", name=account_name)
                    else:
                        raise
        self.accounts = created
        return created

    def create_contacts(self, count_per_account: int = 1) -> List[Dict]:
        created = []
        for account in self.accounts:
            account_id = account.get("account_id")
            account_fields = account.get("fields", {})
            account_name = account_fields.get("name", "Unknown")
            for i in range(count_per_account):
                try:
                    contact_name = f"Contact {i + 1} for {account_name}"
                    contact_email = f"contact{i + 1}.{account_id[:8]}@example.com"
                    response = self._make_request(
                        "POST",
                        "/api/v1/contact/create",
                        json_data={
                            "account_id": account_id,
                            "fields": {
                                "name": contact_name,
                                "description": f"Primary contact for {account_name}",
                                "email": contact_email,
                            },
                        },
                    )
                    response["_account_id"] = account_id
                    response["_agent_type_id"] = account.get("_agent_type_id")
                    created.append(response)
                    logger.info(
                        "Created contact", name=contact_name, account=account_name, id=response.get("contact_id")
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 409:
                        logger.warning("Contact may already exist", name=contact_name)
                    else:
                        raise
        self.contacts = created
        return created

    def create_agent_instances(self) -> List[Dict]:
        created = []
        for account in self.accounts:
            account_id = account.get("account_id")
            account_fields = account.get("fields", {})
            account_name = account_fields.get("name", "Unknown")
            agent_type_id = account.get("_agent_type_id")
            account_contacts = [c for c in self.contacts if c.get("_account_id") == account_id]
            if not account_contacts:
                logger.warning("No contacts found for account", account=account_name)
                continue
            contact_id = account_contacts[0].get("contact_id")
            try:
                agent_name = f"Agent for {account_name}"
                response = self._make_request(
                    "POST",
                    "/api/v1/agent_instance/create",
                    json_data={
                        "name": agent_name,
                        "description": f"AI Agent handling tasks for {account_name}",
                        "account_id": account_id,
                        "contact_id": contact_id,
                        "agent_type_id": agent_type_id,
                    },
                )
                response["_account_id"] = account_id
                response["_agent_type_id"] = agent_type_id
                response["_contacts"] = account_contacts
                response["_anonymous_ids"] = SeededDataManager.generate_anonymous_ids(1, 10)
                created.append(response)
                logger.info(
                    "Created agent instance",
                    name=agent_name,
                    account=account_name,
                    id=response.get("agent_instance_id") or response.get("id"),
                    anonymous_ids_count=len(response["_anonymous_ids"]),
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    logger.warning("Agent instance may already exist", name=agent_name)
                else:
                    raise
        self.agent_instances = created
        return created

    def create_products(self) -> List[Dict]:
        created = []
        for agent_type in self.agent_types:
            agent_type_id = agent_type.get("agent_type_id")
            agent_type_name = agent_type.get("name", "Unknown")
            try:
                product_name = f"{agent_type_name} - Product"
                response = self._make_request(
                    "POST",
                    "/api/v1/products",
                    json_data={
                        "name": product_name,
                        "description": f"Product for {agent_type_name}",
                        "agent_type_id": agent_type_id,
                    },
                )
                response["_agent_type_id"] = agent_type_id
                created.append(response)
                logger.info("Created product", name=product_name, id=response.get("product_id"))
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    logger.warning("Product may already exist, fetching existing", name=product_name)
                    try:
                        products_response = self._make_request("GET", "/api/v1/products")
                        for product in products_response.get("products", []):
                            if product.get("name") == product_name:
                                product["_agent_type_id"] = agent_type_id
                                created.append(product)
                                logger.info("Found existing product", name=product_name, id=product.get("product_id"))
                                break
                    except Exception as fetch_err:
                        logger.error("Failed to fetch existing products", error=str(fetch_err))
                else:
                    raise
        self.products = created
        return created

    def create_rate_plans(self) -> List[Dict]:
        created = []
        for product in self.products:
            product_id = product.get("product_id")
            product_name = product.get("name", "Unknown")
            if not product_id:
                logger.warning("Skipping rate plan - no product_id", product=product_name)
                continue
            try:
                rate_plan_name = f"{RATE_PLAN_CONFIG['name']} - {product_name}"
                product_response = self._make_request(
                    "POST",
                    f"/api/v1/products/{product_id}/rateplans",
                    json_data={
                        "name": rate_plan_name,
                        "description": f"Rate plan for {product_name}",
                        "billing_cycle_code": RATE_PLAN_CONFIG["billing_cycle_code"],
                        "currency_code": RATE_PLAN_CONFIG["currency_code"],
                    },
                )
                rate_plans_list = product_response.get("rate_plans", [])
                if rate_plans_list:
                    rate_plan = rate_plans_list[-1]
                    rate_plan["_product_id"] = product_id
                    rate_plan["_agent_type_id"] = product.get("_agent_type_id")
                    created.append(rate_plan)
                    logger.info("Created rate plan", name=rate_plan_name, id=rate_plan.get("rate_plan_id"))
                else:
                    logger.warning("Rate plan created but not found in response", product=product_name)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    logger.warning("Rate plan may already exist, fetching existing", product=product_name)
                    try:
                        product_response = self._make_request("GET", f"/api/v1/products/{product_id}")
                        rate_plans_list = product_response.get("rate_plans", [])
                        for rate_plan in rate_plans_list:
                            rate_plan["_product_id"] = product_id
                            rate_plan["_agent_type_id"] = product.get("_agent_type_id")
                            created.append(rate_plan)
                            logger.info(
                                "Found existing rate plan", name=rate_plan.get("name"), id=rate_plan.get("rate_plan_id")
                            )
                    except Exception as fetch_err:
                        logger.error("Failed to fetch existing rate plans", error=str(fetch_err))
                else:
                    raise
        self.rate_plans = created
        return created

    def get_charge_models(self) -> List[Dict]:
        try:
            response = self._make_request("GET", "/api/v1/charge_models")
            self.charge_models = response.get("charge_models", [])
            logger.info("Fetched charge models", count=len(self.charge_models))
            return self.charge_models
        except httpx.HTTPStatusError as e:
            logger.warning("Failed to fetch charge models", error=str(e))
            return []

    def _load_charge_models_from_json(self) -> List[Dict]:
        import json
        import os

        json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "charge_models.json")
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("charge_models.json not found, using empty list")
            return []
        except json.JSONDecodeError as e:
            logger.error("Failed to parse charge_models.json", error=str(e))
            return []

    def _load_charge_model_types(self) -> Dict[str, List[str]]:
        import json
        import os

        json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "charge_model_types.json")
        try:
            with open(json_path, "r") as f:
                charge_types = json.load(f)
                return {ct["name"]: ct["supported_model_types"] for ct in charge_types}
        except FileNotFoundError:
            logger.warning("charge_model_types.json not found, using defaults")
            return {
                "one-time": ["flat-fee", "discount"],
                "recurring": ["flat-fee", "discount", "overage-pricing"],
                "usage": ["discount", "per-unit", "tiered", "volume"],
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to parse charge_model_types.json", error=str(e))
            return {}

    def _is_valid_charge_model_for_type(self, charge_type: str, charge_model_type: str) -> bool:
        charge_model_types = self._load_charge_model_types()
        supported_types = charge_model_types.get(charge_type, [])
        return charge_model_type in supported_types

    def _randomize_pricing_strategy(self, charge_model: Dict) -> Dict:
        import copy

        model = copy.deepcopy(charge_model)
        pricing_strategy = model.get("pricing_strategy", {})
        model_type = model.get("type", "")
        discount_types = ["percentage", "flat"]
        unit_of_measures = ["@action", "@outcome", "@seat", "@user"]
        if model_type == "discount" and "discount_type" in pricing_strategy:
            pricing_strategy["discount_type"] = random.choice(discount_types)
        if "unit_of_measure" in pricing_strategy:
            pricing_strategy["unit_of_measure"] = random.choice(unit_of_measures)
        model["pricing_strategy"] = pricing_strategy
        return model

    def create_charge_models(self) -> List[Dict]:
        created = []
        charge_model_configs = self._load_charge_models_from_json()
        if not charge_model_configs:
            logger.warning("No charge models found in charge_models.json")
            return []
        for charge_model_config in charge_model_configs:
            randomized_config = self._randomize_pricing_strategy(charge_model_config)
            try:
                response = self._make_request(
                    "POST",
                    "/api/v1/charge_models",
                    json_data={
                        "name": randomized_config["name"],
                        "description": randomized_config["description"],
                        "type": randomized_config["type"],
                        "pricing_strategy": randomized_config["pricing_strategy"],
                    },
                )
                created.append(response)
                logger.info(
                    "Created charge model",
                    name=randomized_config["name"],
                    type=randomized_config["type"],
                    pricing_strategy=randomized_config["pricing_strategy"],
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    logger.warning("Charge model may already exist", name=randomized_config["name"])
                else:
                    logger.error("Failed to create charge model", name=randomized_config["name"], error=str(e))
        self.get_charge_models()
        return self.charge_models

    def get_charge_model_by_type(self, model_type: str) -> Optional[Dict]:
        for cm in self.charge_models:
            if cm.get("type") == model_type:
                return cm
        return self.charge_models[0] if self.charge_models else None

    def create_rate_plan_charges(self) -> int:
        if not self.charge_models:
            self.get_charge_models()
        if not self.charge_models:
            logger.info("No charge models found, creating them...")
            self.create_charge_models()
        if not self.charge_models:
            logger.warning("No charge models available, skipping rate plan charges")
            return 0
        charge_model_types = self._load_charge_model_types()
        total_charges_created = 0
        for rate_plan in self.rate_plans:
            rate_plan_id = rate_plan.get("rate_plan_id")
            rate_plan_name = rate_plan.get("name", "Unknown")
            if not rate_plan_id:
                logger.warning("Rate plan has no ID, skipping", rate_plan=rate_plan_name)
                continue
            num_charges = random.randint(1, 3)
            selected_charges = random.sample(AVAILABLE_CHARGES, min(num_charges, len(AVAILABLE_CHARGES)))
            for charge_config in selected_charges:
                charge_type = charge_config["charge_type"]
                charge_model_type = charge_config.get("charge_model_type", "flat-fee")
                supported_types = charge_model_types.get(charge_type, [])
                if charge_model_type not in supported_types:
                    logger.warning(
                        "Invalid charge_model_type for charge_type",
                        charge=charge_config["name"],
                        charge_type=charge_type,
                        charge_model_type=charge_model_type,
                        supported_types=supported_types,
                    )
                    continue
                charge_model = self.get_charge_model_by_type(charge_model_type)
                if not charge_model:
                    logger.warning(
                        "No charge model found for charge", charge=charge_config["name"], model_type=charge_model_type
                    )
                    continue
                try:
                    payload = {
                        "name": charge_config["name"],
                        "description": charge_config.get("description", ""),
                        "list_price": charge_config.get("list_price", 0),
                        "charge_type": charge_type,
                        "charge_model_id": charge_model.get("charge_model_id") or charge_model.get("id"),
                        "fields": {},
                    }
                    if charge_config.get("uom"):
                        payload["uom"] = charge_config["uom"]
                    self._make_request("POST", f"/api/v1/products/rateplans/{rate_plan_id}/charges", json_data=payload)
                    total_charges_created += 1
                    logger.info(
                        "Created rate plan charge",
                        rate_plan=rate_plan_name,
                        charge=charge_config["name"],
                        charge_type=charge_type,
                        charge_model_type=charge_model_type,
                        charge_model=charge_model.get("name"),
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 409:
                        logger.warning(
                            "Rate plan charge may already exist", rate_plan=rate_plan_name, charge=charge_config["name"]
                        )
                    else:
                        logger.error(
                            "Failed to create rate plan charge",
                            rate_plan=rate_plan_name,
                            charge=charge_config["name"],
                            error=str(e),
                        )
        logger.info("Rate plan charges creation complete", total_created=total_charges_created)
        return total_charges_created

    def create_subscriptions(self) -> List[Dict]:
        created = []
        if not self.triggers:
            self.get_triggers()
        if not self.triggers:
            logger.warning("No triggers available, skipping subscriptions")
            return created
        default_trigger_id = self.triggers[0].get("trigger_id")
        for account in self.accounts:
            account_id = account.get("account_id")
            account_fields = account.get("fields", {})
            account_name = account_fields.get("name", "Unknown")
            agent_type_id = account.get("_agent_type_id")
            rate_plan = next((rp for rp in self.rate_plans if rp.get("_agent_type_id") == agent_type_id), None)
            if not rate_plan:
                logger.warning("No rate plan found for account", account=account_name, agent_type_id=agent_type_id)
                continue
            rate_plan_id = rate_plan.get("rate_plan_id")
            if not rate_plan_id:
                logger.warning("Rate plan has no ID", rate_plan=rate_plan)
                continue
            try:
                activation_time_delta = timedelta(days=random.randint(3, self.seed_months * 30))
                desired_activated_at = datetime.now(timezone.utc) - activation_time_delta
                response = self._make_request(
                    "POST",
                    "/api/v1/subscription/create",
                    json_data={
                        "name": f"Subscription - {account_name}",
                        "account_id": account_id,
                        "rate_plan_id": rate_plan_id,
                        "trigger_id": default_trigger_id,
                    },
                )
                subscription_id = response.get("subscription_id")
                if subscription_id:
                    self._update_subscription_activated_at(subscription_id, desired_activated_at)
                response["_account_id"] = account_id
                response["_agent_type_id"] = agent_type_id
                created.append(response)
                logger.info("Created subscription", account=account_name, id=subscription_id)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    logger.warning("Subscription may already exist", account=account_name)
                else:
                    raise
        self.subscriptions = created
        return created

    def create_cost_allocation_rules(self) -> Dict:
        try:
            action_costs = [
                {
                    "action_value": rule["action_value"],
                    "action_type": rule["action_type"],
                    "measure": rule["measure"],
                    "cost": rule["cost"],
                }
                for rule in COST_ALLOCATION_RULES
            ]
            response = self._make_request("PUT", "/api/v1/action-costs", json_data={"action_costs": action_costs})
            logger.info("Updated cost allocation rules", count=len(action_costs))
            return response
        except httpx.HTTPStatusError as e:
            logger.warning("Failed to update cost allocation rules", error=str(e))
            raise

    def update_tagging_config(self) -> Dict:
        try:
            auto_tags = [tag for tag in TAGGED_ACTIONS if tag.startswith("gemini_") or tag.startswith("langchain_")]
            trace_events = [tag for tag in TAGGED_ACTIONS if tag not in auto_tags]
            self._make_request(
                "PUT",
                "/api/v1/tagged_actions",
                json_data={
                    "tagged_actions": {
                        "trace_events": trace_events,
                        "auto_tags": auto_tags if auto_tags else TAGGED_ACTIONS,
                    }
                },
            )
            logger.info("Updated tagged actions", auto_tags=auto_tags, trace_events=trace_events)
            self._make_request(
                "PUT",
                "/api/v1/tagged_outcomes",
                json_data={
                    "tagged_actions": {
                        "trace_events": TAGGED_OUTCOMES.get("trace_events", []),
                        "auto_tags": TAGGED_OUTCOMES.get("auto_tags", []),
                    }
                },
            )
            logger.info(
                "Updated tagged outcomes",
                trace_events=TAGGED_OUTCOMES.get("trace_events", []),
                auto_tags=TAGGED_OUTCOMES.get("auto_tags", []),
            )
            return {"status": "success"}
        except Exception as e:
            logger.warning("Failed to update tagging config", error=str(e))
            return {"status": "partial", "error": str(e)}

    def _load_invoice_templates_from_json(self) -> List[Dict]:
        import json
        import os

        json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "default_invoice_template.json")
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("default_invoice_template.json not found, using empty list")
            return []
        except json.JSONDecodeError as e:
            logger.error("Failed to parse default_invoice_template.json", error=str(e))
            return []

    def create_invoice_template(self) -> Optional[Dict]:
        templates = self._load_invoice_templates_from_json()
        if not templates:
            logger.warning("No invoice templates found in default_invoice_template.json")
            return None
        created_template = None
        for template_config in templates:
            try:
                payload = {
                    "name": template_config["name"],
                    "description": template_config.get("description", ""),
                    "markdown_content": template_config["markdown_content"],
                    "css_content": template_config["css_content"],
                    "html_content": template_config["html_content"],
                    "set_as_active": template_config.get("is_active", False),
                }
                response = self._make_request("POST", "/api/v1/invoice/templates/create", json_data=payload)
                created_template = response
                logger.info(
                    "Created invoice template",
                    name=template_config["name"],
                    is_active=template_config.get("is_active", False),
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    logger.warning("Invoice template may already exist", name=template_config["name"])
                else:
                    logger.error(
                        "Failed to create invoice template",
                        name=template_config["name"],
                        error=str(e),
                        status_code=e.response.status_code,
                    )
            except Exception as e:
                logger.error(
                    "Unexpected error creating invoice template",
                    name=template_config.get("name", "unknown"),
                    error=str(e),
                )
        return created_template

    def seed_all(self) -> Dict[str, Any]:
        logger.info("Starting Control Plane seeding...")
        self.get_current_workspace()
        self.create_agent_types()
        logger.info("Agent types created/found", count=len(self.agent_types))
        self.create_accounts(count_per_agent_type=5)
        logger.info("Accounts created", count=len(self.accounts))
        self.create_contacts(count_per_account=1)
        logger.info("Contacts created", count=len(self.contacts))
        self.create_agent_instances()
        logger.info("Agent instances created", count=len(self.agent_instances))
        self.create_products()
        logger.info("Products created", count=len(self.products))
        if self.products:
            self.create_rate_plans()
            logger.info("Rate plans created", count=len(self.rate_plans))
            if self.rate_plans:
                charges_created = self.create_rate_plan_charges()
                logger.info("Rate plan charges created", count=charges_created)
        else:
            logger.warning("Skipping rate plans - no products available")
        if self.rate_plans:
            self.create_subscriptions()
            logger.info("Subscriptions created", count=len(self.subscriptions))
        else:
            logger.warning("Skipping subscriptions - no rate plans available")
        try:
            self.create_cost_allocation_rules()
        except Exception as e:
            logger.warning("Failed to create cost allocation rules", error=str(e))
        try:
            self.update_tagging_config()
        except Exception as e:
            logger.warning("Failed to update tagging config", error=str(e))
        invoice_template_created = False
        try:
            template = self.create_invoice_template()
            invoice_template_created = template is not None
            if invoice_template_created:
                logger.info("Invoice template created")
        except Exception as e:
            logger.warning("Failed to create invoice template", error=str(e))
        summary = {
            "workspace_id": self.workspace_id,
            "organization_id": self.organization_id,
            "agent_types": len(self.agent_types),
            "accounts": len(self.accounts),
            "contacts": len(self.contacts),
            "agent_instances": len(self.agent_instances),
            "products": len(self.products),
            "rate_plans": len(self.rate_plans),
            "subscriptions": len(self.subscriptions),
            "invoice_template_created": invoice_template_created,
        }
        logger.info("Control Plane seeding complete", **summary)
        return summary

    def get_agent_data_for_traces(self) -> List[Dict]:
        agents_data = []
        for agent in self.agent_instances:
            anonymous_ids = agent.get("_anonymous_ids", [])
            if not anonymous_ids:
                anonymous_ids = SeededDataManager.generate_anonymous_ids(1, 10)
            agents_data.append(
                {
                    "agent_id": agent.get("id"),
                    "agent_name": agent.get("name"),
                    "account_id": agent.get("_account_id"),
                    "workspace_id": self.workspace_id,
                    "organization_id": self.organization_id,
                    "anonymous_ids": anonymous_ids,
                }
            )
        return agents_data

    def save_seeded_data(self, filepath: Optional[str] = None) -> bool:
        manager = SeededDataManager(filepath)
        agents_data = self.get_agent_data_for_traces()
        manager.set_data(workspace_id=self.workspace_id, organization_id=self.organization_id, agents=agents_data)
        return manager.save()

    def refresh_agent_secrets(self, agent_ids: Optional[List[str]] = None) -> Dict[str, str]:
        if agent_ids is None:
            agent_ids = [agent.get("id") for agent in self.agent_instances]
        secrets = {}
        for agent_id in agent_ids:
            try:
                response = self._make_request("POST", f"/api/v1/agent_instance/{agent_id}/refresh")
                secret = response.get("secret_token") or response.get("secret")
                if secret:
                    secrets[agent_id] = secret
                    logger.info("Refreshed agent secret", agent_id=agent_id)
            except httpx.HTTPStatusError as e:
                logger.error("Failed to refresh agent secret", agent_id=agent_id, status_code=e.response.status_code)
            except Exception as e:
                logger.error("Failed to refresh agent secret", agent_id=agent_id, error=str(e))
        return secrets

    def close(self):
        self.client.close()
        if self._postgres_conn and not self._postgres_conn.closed:
            self._postgres_conn.close()
            self._postgres_conn = None
