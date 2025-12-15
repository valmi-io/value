"""ClickHouse seeder for trace data."""

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import structlog

from .config import (
    CLICKHOUSE_DB,
    CLICKHOUSE_HOST,
    CLICKHOUSE_HTTP_PORT,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_USER,
    SEED_EVENTS_PER_DAY,
    SEED_MONTHS,
)
from .data_generators import generate_events_count_for_day, generate_pipeline_traces
from .seeded_data import SeededDataManager

logger = structlog.get_logger(__name__)

TRACES_TABLE = "scratch_traces_flatten_measure_table"

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {database}.{table} (
    id UUID DEFAULT generateUUIDv7(),
    created_at DateTime64(3) DEFAULT now(),
    message_id String,
    trace_created_at DateTime64(3),
    resourceSpans_scopeSpans_spans_traceId Nullable(String),
    resourceSpans_scopeSpans_spans_spanId Nullable(String),
    resourceSpans_scopeSpans_spans_parentSpanId Nullable(String),
    resourceSpans_scopeSpans_spans_kind Nullable(Int64),
    resourceSpans_scopeSpans_spans_startTimeUnixNano Nullable(DateTime64(9)),
    resourceSpans_scopeSpans_spans_endTimeUnixNano Nullable(DateTime64(9)),
    resourceSpans_resource_attributes_service_name Nullable(String),
    resourceSpans_resource_attributes_service_version Nullable(String),
    resourceSpans_resource_attributes_value_client_sdk Nullable(String),
    resourceSpans_resource_attributes_value_agent_organization_id Nullable(String),
    resourceSpans_resource_attributes_value_agent_workspace_id Nullable(String),
    resourceSpans_resource_attributes_value_agent_name Nullable(String),
    resourceSpans_resource_attributes_value_agent_id Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_name Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_user_id Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_anonymous_id Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_description Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_type Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_status Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_error Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_duration Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_start_time Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_end_time Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_llm_model Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_llm_input_tokens Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_llm_output_tokens Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_llm_total_tokens Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_llm_prompt Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_llm_response Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_value_action_user_attributes Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_gen_ai_request_model Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_gen_ai_response_model Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_prompt_tokens Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_completion_tokens Nullable(String),
    resourceSpans_scopeSpans_spans_attributes_llm_usage_total_tokens Nullable(String),
    resourceSpans_scopeSpans_spans_events_name Nullable(String),
    resourceSpans_scopeSpans_spans_events_timeUnixNano Nullable(DateTime64(9)),
    tag Nullable(String),
    model Nullable(String),
    resourceSpans_scopeSpans_spans_name Nullable(String),
    measure Nullable(String)
) ENGINE = ReplacingMergeTree()
ORDER BY (created_at, id)
"""


class ClickHouseSeeder:
    def __init__(self):
        self.client = None
        self.database = CLICKHOUSE_DB

    def connect(self):
        try:
            from clickhouse_connect import get_client

            self.client = get_client(
                host=CLICKHOUSE_HOST,
                port=CLICKHOUSE_HTTP_PORT,
                database=CLICKHOUSE_DB,
                username=CLICKHOUSE_USER,
                password=CLICKHOUSE_PASSWORD if CLICKHOUSE_PASSWORD else None,
            )
            logger.info(
                "Connected to ClickHouse", host=CLICKHOUSE_HOST, port=CLICKHOUSE_HTTP_PORT, database=CLICKHOUSE_DB
            )
        except Exception as e:
            logger.error("Failed to connect to ClickHouse", error=str(e))
            raise

    def clear_table(self):
        try:
            self.client.command(f"TRUNCATE TABLE IF EXISTS {self.database}.{TRACES_TABLE}")
            logger.info("Cleared table", table=TRACES_TABLE)
        except Exception as e:
            logger.warning("Failed to truncate table, may not exist", error=str(e))

    def ensure_table_exists(self):
        try:
            schema = TABLE_SCHEMA.format(database=self.database, table=TRACES_TABLE)
            self.client.command(schema)
            logger.info("Ensured table exists", table=TRACES_TABLE)
        except Exception as e:
            logger.error("Failed to create table", error=str(e))
            raise

    def insert_traces(self, traces: List[Dict[str, Any]]):
        if not traces:
            return
        columns = [
            "id",
            "created_at",
            "message_id",
            "trace_created_at",
            "resourceSpans_scopeSpans_spans_traceId",
            "resourceSpans_scopeSpans_spans_spanId",
            "resourceSpans_scopeSpans_spans_parentSpanId",
            "resourceSpans_scopeSpans_spans_kind",
            "resourceSpans_scopeSpans_spans_startTimeUnixNano",
            "resourceSpans_scopeSpans_spans_endTimeUnixNano",
            "resourceSpans_resource_attributes_service_name",
            "resourceSpans_resource_attributes_service_version",
            "resourceSpans_resource_attributes_value_client_sdk",
            "resourceSpans_resource_attributes_value_agent_organization_id",
            "resourceSpans_resource_attributes_value_agent_workspace_id",
            "resourceSpans_resource_attributes_value_agent_name",
            "resourceSpans_resource_attributes_value_agent_id",
            "resourceSpans_scopeSpans_spans_attributes_value_action_name",
            "resourceSpans_scopeSpans_spans_attributes_value_action_user_id",
            "resourceSpans_scopeSpans_spans_attributes_value_action_anonymous_id",
            "resourceSpans_scopeSpans_spans_attributes_value_action_description",
            "resourceSpans_scopeSpans_spans_attributes_value_action_type",
            "resourceSpans_scopeSpans_spans_attributes_value_action_status",
            "resourceSpans_scopeSpans_spans_attributes_value_action_error",
            "resourceSpans_scopeSpans_spans_attributes_value_action_duration",
            "resourceSpans_scopeSpans_spans_attributes_value_action_start_time",
            "resourceSpans_scopeSpans_spans_attributes_value_action_end_time",
            "resourceSpans_scopeSpans_spans_attributes_value_action_llm_model",
            "resourceSpans_scopeSpans_spans_attributes_value_action_llm_input_tokens",
            "resourceSpans_scopeSpans_spans_attributes_value_action_llm_output_tokens",
            "resourceSpans_scopeSpans_spans_attributes_value_action_llm_total_tokens",
            "resourceSpans_scopeSpans_spans_attributes_value_action_llm_prompt",
            "resourceSpans_scopeSpans_spans_attributes_value_action_llm_response",
            "resourceSpans_scopeSpans_spans_attributes_value_action_user_attributes",
            "resourceSpans_scopeSpans_spans_attributes_gen_ai_request_model",
            "resourceSpans_scopeSpans_spans_attributes_gen_ai_response_model",
            "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_prompt_tokens",
            "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_completion_tokens",
            "resourceSpans_scopeSpans_spans_attributes_llm_usage_total_tokens",
            "resourceSpans_scopeSpans_spans_events_name",
            "resourceSpans_scopeSpans_spans_events_timeUnixNano",
            "tag",
            "model",
            "resourceSpans_scopeSpans_spans_name",
            "measure",
        ]
        rows = []
        for trace in traces:
            row = []
            for col in columns:
                value = trace.get(col)
                if col == "id":
                    row.append(value)
                elif col in ("created_at", "trace_created_at"):
                    row.append(datetime.fromtimestamp(value / 1000, tz=timezone.utc) if value else None)
                elif col.endswith("TimeUnixNano"):
                    row.append(datetime.fromtimestamp(value / 1e9, tz=timezone.utc) if value else None)
                else:
                    row.append(value)
            rows.append(row)
        try:
            self.client.insert(table=TRACES_TABLE, data=rows, column_names=columns)
            logger.debug("Inserted traces", count=len(traces))
        except Exception as e:
            logger.error("Failed to insert traces", error=str(e), count=len(traces))
            raise

    def seed_past_data(
        self, agents_data: List[Dict], months: int = None, events_per_day: int = None, clear_existing: bool = True
    ) -> Dict[str, Any]:
        months = months or SEED_MONTHS
        events_per_day = events_per_day or SEED_EVENTS_PER_DAY
        logger.info("Starting historical data seeding", months=months, base_events_per_day=events_per_day)
        if clear_existing:
            self.clear_table()
        self.ensure_table_exists()
        if not agents_data:
            logger.warning("No agents data provided, using defaults")
            agents_data = self._create_default_agents()
        end_date = datetime.now(timezone.utc) - timedelta(days=1)
        start_date = end_date - timedelta(days=months * 30)
        total_traces = 0
        total_days = 0
        current_date = start_date
        batch_size = 1000
        batch = []
        while current_date <= end_date:
            day_events = generate_events_count_for_day(events_per_day, current_date)
            for _ in range(day_events):
                hour = random.choices(
                    range(24),
                    weights=[
                        0.5,
                        0.3,
                        0.2,
                        0.2,
                        0.3,
                        0.5,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        4.0,
                        3.0,
                        2.5,
                        3.0,
                        4.0,
                        4.0,
                        3.0,
                        2.0,
                        1.5,
                        1.0,
                        0.8,
                        0.6,
                        0.5,
                        0.4,
                    ],
                )[0]
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                event_time = current_date.replace(
                    hour=hour, minute=minute, second=second, microsecond=random.randint(0, 999999)
                )
                agent = random.choice(agents_data)
                traces = generate_pipeline_traces(
                    timestamp=event_time,
                    agent_id=agent["agent_id"],
                    agent_name=agent["agent_name"],
                    workspace_id=agent["workspace_id"],
                    organization_id=agent["organization_id"],
                    anonymous_ids=agent["anonymous_ids"],
                )
                batch.extend(traces)
                total_traces += len(traces)
                if len(batch) >= batch_size:
                    self.insert_traces(batch)
                    batch = []
            total_days += 1
            if total_days % 7 == 0:
                logger.info(
                    "Seeding progress",
                    days_processed=total_days,
                    traces_generated=total_traces,
                    current_date=current_date.strftime("%Y-%m-%d"),
                )
            current_date += timedelta(days=1)
        if batch:
            self.insert_traces(batch)
        summary = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_days": total_days,
            "total_traces": total_traces,
            "agents_used": len(agents_data),
        }
        logger.info("Historical data seeding complete", **summary)
        return summary

    def _create_default_agents(self) -> List[Dict]:
        agents = []
        for i in range(10):
            anonymous_ids = SeededDataManager.generate_anonymous_ids(1, 10)
            agents.append(
                {
                    "agent_id": f"agent_default_{i}",
                    "agent_name": f"Default Agent {i + 1}",
                    "account_id": f"acc_default_{i}",
                    "workspace_id": "ws_default",
                    "organization_id": "org_default",
                    "anonymous_ids": anonymous_ids,
                }
            )
        return agents

    def get_row_count(self) -> int:
        try:
            result = self.client.query(f"SELECT count() FROM {self.database}.{TRACES_TABLE}")
            return result.result_rows[0][0] if result.result_rows else 0
        except Exception:
            return 0

    def table_exists(self) -> bool:
        try:
            result = self.client.query(f"EXISTS TABLE {self.database}.{TRACES_TABLE}")
            return result.result_rows[0][0] == 1 if result.result_rows else False
        except Exception:
            return False

    def seed_live_metrics(self) -> Dict[str, Any]:
        live_metrics_table = "live_metrics"
        logger.info("Starting live_metrics seeding")
        try:
            self.client.command(f"TRUNCATE TABLE IF EXISTS {self.database}.{live_metrics_table}")
            logger.info("Cleared live_metrics table")
        except Exception as e:
            logger.warning("Failed to truncate live_metrics table, may not exist", error=str(e))
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.database}.{live_metrics_table} (
            agent_id String, workspace_id String, organization_id String, metric_date Date,
            action_count UInt64, outcome_count UInt64, user_count UInt64,
            action_data String, outcome_data String, updated_at DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(updated_at)
        ORDER BY (agent_id, workspace_id, organization_id, metric_date)
        PRIMARY KEY (agent_id, workspace_id, organization_id, metric_date)
        """
        self.client.command(create_table_query)
        logger.info("Ensured live_metrics table exists")
        dates_query = f"SELECT DISTINCT toDate(created_at) as metric_date FROM {self.database}.{TRACES_TABLE} ORDER BY metric_date"
        dates_result = self.client.query(dates_query)
        dates = [row[0] for row in dates_result.result_rows]
        if not dates:
            logger.warning("No dates found in measure table")
            return {"total_days": 0, "total_rows": 0}
        logger.info(f"Found {len(dates)} unique dates to process", start_date=str(dates[0]), end_date=str(dates[-1]))
        total_rows = 0
        for date in dates:
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)
            insert_query = f"""
            INSERT INTO {self.database}.{live_metrics_table}
            SELECT agent_id, workspace_id, organization_id, metric_date, action_count, outcome_count, user_count, action_data, outcome_data, updated_at
            FROM (
                SELECT
                    resourceSpans_resource_attributes_value_agent_id as agent_id,
                    resourceSpans_resource_attributes_value_agent_workspace_id as workspace_id,
                    resourceSpans_resource_attributes_value_agent_organization_id as organization_id,
                    toDate(created_at) as metric_date,
                    countIf(resourceSpans_scopeSpans_spans_name = 'value.action' AND (measure != '@outcome' OR measure IS NULL OR measure = '')) as action_count,
                    countIf(resourceSpans_scopeSpans_spans_name = 'value.action' AND measure = '@outcome') as outcome_count,
                    uniqExact(resourceSpans_scopeSpans_spans_attributes_value_action_anonymous_id) as user_count,
                    arrayStringConcat(arrayMap(x -> concat('{{"name":"', x.1, '","count":', toString(x.2), '}}'), arrayZip(sumMapIf([resourceSpans_scopeSpans_spans_attributes_value_action_name], [toUInt64(1)], resourceSpans_scopeSpans_spans_name = 'value.action' AND (measure != '@outcome' OR measure IS NULL OR measure = '') AND resourceSpans_scopeSpans_spans_attributes_value_action_name != '').1, sumMapIf([resourceSpans_scopeSpans_spans_attributes_value_action_name], [toUInt64(1)], resourceSpans_scopeSpans_spans_name = 'value.action' AND (measure != '@outcome' OR measure IS NULL OR measure = '') AND resourceSpans_scopeSpans_spans_attributes_value_action_name != '').2)), ',') as action_data_array,
                    concat('[', action_data_array, ']') as action_data,
                    arrayStringConcat(arrayMap(x -> concat('{{"name":"', x.1, '","count":', toString(x.2), '}}'), arrayZip(sumMapIf([resourceSpans_scopeSpans_spans_attributes_value_action_name], [toUInt64(1)], resourceSpans_scopeSpans_spans_name = 'value.action' AND measure = '@outcome' AND resourceSpans_scopeSpans_spans_attributes_value_action_name != '').1, sumMapIf([resourceSpans_scopeSpans_spans_attributes_value_action_name], [toUInt64(1)], resourceSpans_scopeSpans_spans_name = 'value.action' AND measure = '@outcome' AND resourceSpans_scopeSpans_spans_attributes_value_action_name != '').2)), ',') as outcome_data_array,
                    concat('[', outcome_data_array, ']') as outcome_data,
                    now() as updated_at
                FROM {self.database}.{TRACES_TABLE}
                WHERE toDate(created_at) = '{date_str}'
                GROUP BY agent_id, workspace_id, organization_id, metric_date
            )
            """
            try:
                self.client.command(insert_query)
                count_result = self.client.query(
                    f"SELECT count() FROM {self.database}.{live_metrics_table} WHERE metric_date = '{date_str}'"
                )
                rows_for_date = count_result.result_rows[0][0] if count_result.result_rows else 0
                total_rows += rows_for_date
                logger.debug(f"Processed date {date_str}", rows=rows_for_date)
            except Exception as e:
                logger.error(f"Failed to process date {date_str}", error=str(e))
        final_count_result = self.client.query(f"SELECT count() FROM {self.database}.{live_metrics_table}")
        final_count = final_count_result.result_rows[0][0] if final_count_result.result_rows else 0
        summary = {
            "total_days": len(dates),
            "total_rows": final_count,
            "start_date": str(dates[0]) if dates else None,
            "end_date": str(dates[-1]) if dates else None,
        }
        logger.info("Live metrics seeding complete", **summary)
        return summary

    def get_live_metrics_count(self) -> int:
        try:
            result = self.client.query(f"SELECT count() FROM {self.database}.live_metrics")
            return result.result_rows[0][0] if result.result_rows else 0
        except Exception:
            return 0

    def close(self):
        if self.client:
            try:
                self.client.close()
                logger.info("Closed ClickHouse connection")
            except Exception:
                pass
