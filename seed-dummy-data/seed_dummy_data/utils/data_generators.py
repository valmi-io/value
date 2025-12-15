"""Data generators for seed data creation."""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .config import INVOICE_TYPES


def generate_trace_id() -> str:
    return hex(random.getrandbits(128))[2:].zfill(32)


def generate_span_id() -> str:
    return hex(random.getrandbits(64))[2:].zfill(16)


def generate_message_id() -> str:
    return hex(random.getrandbits(64))[2:].zfill(16)


def generate_uuid7() -> str:
    try:
        from uuid_extensions import uuid7

        return str(uuid7())
    except ImportError:
        return str(uuid.uuid4())


def generate_user_id() -> str:
    return f"user_{random.randint(1000, 9999)}"


def generate_anonymous_id() -> str:
    return str(uuid.uuid4())


def generate_invoice_id() -> str:
    return f"INV-{random.randint(10000, 99999)}"


def random_weekday_weight(date: datetime) -> float:
    weekday = date.weekday()
    if weekday < 5:
        return random.uniform(0.8, 1.2)
    else:
        return random.uniform(0.3, 0.6)


def generate_events_count_for_day(base_count: int, date: datetime) -> int:
    weight = random_weekday_weight(date)
    variation = random.uniform(0.7, 1.3)
    count = int(base_count * weight * variation)
    return max(10, count)


def generate_pipeline_traces(
    timestamp: datetime,
    agent_id: str,
    agent_name: str,
    workspace_id: str,
    organization_id: str,
    anonymous_ids: List[str],
) -> List[Dict[str, Any]]:
    traces = []
    contact_anonymous_id = random.choice(anonymous_ids) if anonymous_ids else generate_anonymous_id()
    contact_user_id = f"user_{contact_anonymous_id[:8]}"
    invoice_id = generate_invoice_id()
    message_id = generate_message_id()
    current_time = timestamp

    base_attrs = {
        "resourceSpans_resource_attributes_service_name": "value-control-agent",
        "resourceSpans_resource_attributes_service_version": "0.1.0",
        "resourceSpans_resource_attributes_value_client_sdk": "value-python",
        "resourceSpans_resource_attributes_value_agent_organization_id": organization_id,
        "resourceSpans_resource_attributes_value_agent_workspace_id": workspace_id,
        "resourceSpans_resource_attributes_value_agent_name": agent_name,
        "resourceSpans_resource_attributes_value_agent_id": agent_id,
    }

    # Object Detection
    duration_ms = random.uniform(500, 1500)
    start_time = current_time
    end_time = start_time + timedelta(milliseconds=duration_ms)
    traces.append(
        create_action_trace(
            action_name="object_detection",
            user_id=contact_user_id,
            anonymous_id=contact_anonymous_id,
            description=f"Detected {random.randint(3, 10)} objects in {invoice_id}",
            user_attributes={
                "invoice_id": invoice_id,
                "model": "yolov8-invoice",
                "confidence_score": round(random.uniform(0.85, 0.99), 4),
                "detected_boxes": random.randint(3, 10),
                "processing_device": "cuda:0",
            },
            start_time=start_time,
            end_time=end_time,
            message_id=message_id,
            base_attrs=base_attrs,
        )
    )
    current_time = end_time + timedelta(milliseconds=random.uniform(100, 500))

    # Object Classification
    duration_ms = random.uniform(300, 1000)
    start_time = current_time
    end_time = start_time + timedelta(milliseconds=duration_ms)
    doc_type = random.choice(INVOICE_TYPES)
    traces.append(
        create_action_trace(
            action_name="object_classification",
            user_id=contact_user_id,
            anonymous_id=contact_anonymous_id,
            description=f"Classified document as {doc_type}",
            user_attributes={
                "invoice_id": invoice_id,
                "document_type": doc_type,
                "classification_confidence": round(random.uniform(0.90, 0.99), 4),
                "model": "resnet50-finetuned",
            },
            start_time=start_time,
            end_time=end_time,
            message_id=message_id,
            base_attrs=base_attrs,
        )
    )
    current_time = end_time + timedelta(milliseconds=random.uniform(100, 500))

    # OCR Processing
    duration_ms = random.uniform(500, 2000)
    start_time = current_time
    end_time = start_time + timedelta(milliseconds=duration_ms)
    traces.append(
        create_action_trace(
            action_name="ocr_processing",
            user_id=contact_user_id,
            anonymous_id=contact_anonymous_id,
            description=f"Extracted text from {invoice_id}",
            user_attributes={
                "invoice_id": invoice_id,
                "ocr_engine": "tesseract",
                "language": "eng",
                "text_length": random.randint(50, 60),
                "confidence": round(random.uniform(0.80, 0.95), 4),
            },
            start_time=start_time,
            end_time=end_time,
            message_id=message_id,
            base_attrs=base_attrs,
            measure="@outcome",
        )
    )
    current_time = end_time + timedelta(milliseconds=random.uniform(100, 500))

    # RAG Retrieval
    duration_ms = random.uniform(200, 800)
    start_time = current_time
    end_time = start_time + timedelta(milliseconds=duration_ms)
    chunks_count = random.randint(2, 5)
    traces.append(
        create_action_trace(
            action_name="rag_retrieval",
            user_id=contact_user_id,
            anonymous_id=contact_anonymous_id,
            description=f"Retrieved {chunks_count} chunks for context",
            user_attributes={
                "invoice_id": invoice_id,
                "vector_db": "chroma",
                "embedding_model": "text-embedding-3-small",
                "chunks_count": chunks_count,
                "top_k": 5,
            },
            start_time=start_time,
            end_time=end_time,
            message_id=message_id,
            base_attrs=base_attrs,
        )
    )
    current_time = end_time + timedelta(milliseconds=random.uniform(100, 500))

    # LLM Call + Response Processing
    is_gemini = random.choice([True, False])
    duration_ms = random.uniform(1000, 2500)
    start_time = current_time
    end_time = start_time + timedelta(milliseconds=duration_ms)

    if is_gemini:
        model = "gemini-2.5-flash"
        traces.append(
            create_llm_trace(
                model=model,
                prompt_tokens=random.randint(10, 50),
                completion_tokens=random.randint(50, 200),
                total_tokens=random.randint(100, 500),
                user_id=contact_user_id,
                anonymous_id=contact_anonymous_id,
                start_time=start_time,
                end_time=end_time,
                message_id=message_id,
                base_attrs=base_attrs,
                is_gemini=True,
            )
        )
        traces.append(
            create_action_trace(
                action_name="process_gemini_response",
                user_id=contact_user_id,
                anonymous_id=contact_anonymous_id,
                description=f"Received response from {model}",
                user_attributes={
                    "custom.model": model,
                    "custom.response_length": 51,
                    "custom.prompt_type": "summarization",
                },
                start_time=start_time,
                end_time=start_time + timedelta(milliseconds=50),
                message_id=message_id,
                base_attrs=base_attrs,
                measure="@outcome",
            )
        )
    else:
        model = "llama3"
        traces.append(
            create_llm_trace(
                model=model,
                prompt_tokens=random.randint(10, 50),
                completion_tokens=random.randint(50, 200),
                total_tokens=random.randint(100, 600),
                user_id=None,
                anonymous_id=None,
                start_time=start_time,
                end_time=end_time,
                message_id=message_id,
                base_attrs=base_attrs,
                is_gemini=False,
                invoice_id=invoice_id,
            )
        )
        traces.append(
            create_action_trace(
                action_name="process_ollama_response",
                user_id=contact_user_id,
                anonymous_id=contact_anonymous_id,
                description=f"Received response from {model}",
                user_attributes={
                    "custom.model": model,
                    "custom.response_length": 45,
                    "custom.prompt_type": "extraction",
                },
                start_time=start_time,
                end_time=start_time + timedelta(milliseconds=50),
                message_id=message_id,
                base_attrs=base_attrs,
                measure="@outcome",
            )
        )

    return traces


def create_action_trace(
    action_name: str,
    user_id: str,
    anonymous_id: str,
    description: str,
    user_attributes: Dict[str, Any],
    start_time: datetime,
    end_time: datetime,
    message_id: str,
    base_attrs: Dict[str, Any],
    measure: str | None = None,
) -> Dict[str, Any]:
    import json

    start_ns = start_time.timestamp() * 1e9
    end_ns = end_time.timestamp() * 1e9
    trace = {
        "id": generate_uuid7(),
        "created_at": int(end_time.timestamp() * 1000),
        "message_id": message_id,
        "trace_created_at": int(start_time.timestamp() * 1000),
        "resourceSpans_scopeSpans_spans_traceId": generate_trace_id(),
        "resourceSpans_scopeSpans_spans_spanId": generate_span_id(),
        "resourceSpans_scopeSpans_spans_parentSpanId": None,
        "resourceSpans_scopeSpans_spans_kind": 1,
        "resourceSpans_scopeSpans_spans_startTimeUnixNano": start_ns,
        "resourceSpans_scopeSpans_spans_endTimeUnixNano": end_ns,
        "resourceSpans_scopeSpans_spans_attributes_value_action_name": action_name,
        "resourceSpans_scopeSpans_spans_attributes_value_action_user_id": user_id,
        "resourceSpans_scopeSpans_spans_attributes_value_action_anonymous_id": anonymous_id,
        "resourceSpans_scopeSpans_spans_attributes_value_action_description": description,
        "resourceSpans_scopeSpans_spans_attributes_value_action_type": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_status": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_error": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_duration": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_start_time": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_end_time": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_llm_model": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_llm_input_tokens": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_llm_output_tokens": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_llm_total_tokens": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_llm_prompt": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_llm_response": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_user_attributes": json.dumps(user_attributes),
        "resourceSpans_scopeSpans_spans_attributes_gen_ai_request_model": None,
        "resourceSpans_scopeSpans_spans_attributes_gen_ai_response_model": None,
        "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_prompt_tokens": None,
        "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_completion_tokens": None,
        "resourceSpans_scopeSpans_spans_attributes_llm_usage_total_tokens": None,
        "resourceSpans_scopeSpans_spans_events_name": None,
        "resourceSpans_scopeSpans_spans_events_timeUnixNano": None,
        "tag": None,
        "model": None,
        "resourceSpans_scopeSpans_spans_name": "value.action",
        "measure": measure,
        **base_attrs,
    }
    return trace


def create_llm_trace(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    user_id: str | None,
    anonymous_id: str | None,
    start_time: datetime,
    end_time: datetime,
    message_id: str,
    base_attrs: Dict[str, Any],
    is_gemini: bool = True,
    invoice_id: str = None,
) -> Dict[str, Any]:
    start_ns = start_time.timestamp() * 1e9
    end_ns = end_time.timestamp() * 1e9
    trace = {
        "id": generate_uuid7(),
        "created_at": int(end_time.timestamp() * 1000),
        "message_id": message_id,
        "trace_created_at": int(start_time.timestamp() * 1000),
        "resourceSpans_scopeSpans_spans_traceId": generate_trace_id(),
        "resourceSpans_scopeSpans_spans_spanId": generate_span_id(),
        "resourceSpans_scopeSpans_spans_parentSpanId": None,
        "resourceSpans_scopeSpans_spans_kind": 3,
        "resourceSpans_scopeSpans_spans_startTimeUnixNano": start_ns,
        "resourceSpans_scopeSpans_spans_endTimeUnixNano": end_ns,
        "resourceSpans_scopeSpans_spans_attributes_value_action_name": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_user_id": user_id,
        "resourceSpans_scopeSpans_spans_attributes_value_action_anonymous_id": anonymous_id,
        "resourceSpans_scopeSpans_spans_attributes_value_action_description": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_type": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_status": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_error": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_duration": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_start_time": None,
        "resourceSpans_scopeSpans_spans_attributes_value_action_end_time": None,
        "resourceSpans_scopeSpans_spans_events_name": None,
        "resourceSpans_scopeSpans_spans_events_timeUnixNano": None,
        "resourceSpans_scopeSpans_spans_name": "value.action",
        "measure": None,
        **base_attrs,
    }
    if is_gemini:
        trace.update(
            {
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_model": None,
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_input_tokens": None,
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_output_tokens": None,
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_total_tokens": None,
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_prompt": None,
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_response": None,
                "resourceSpans_scopeSpans_spans_attributes_value_action_user_attributes": None,
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_request_model": model,
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_response_model": model,
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_prompt_tokens": str(prompt_tokens),
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_completion_tokens": str(completion_tokens),
                "resourceSpans_scopeSpans_spans_attributes_llm_usage_total_tokens": str(total_tokens),
                "tag": f"gemini_llm_{model}",
                "model": model,
            }
        )
    else:
        trace.update(
            {
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_model": model,
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_input_tokens": str(prompt_tokens),
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_output_tokens": str(completion_tokens),
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_total_tokens": str(total_tokens),
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_prompt": (
                    f"Extract details from {invoice_id}" if invoice_id else "Process request"
                ),
                "resourceSpans_scopeSpans_spans_attributes_value_action_llm_response": (
                    f"Extracted details for {invoice_id} using Ollama." if invoice_id else "Response from Ollama."
                ),
                "resourceSpans_scopeSpans_spans_attributes_value_action_user_attributes": None,
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_request_model": None,
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_response_model": None,
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_prompt_tokens": None,
                "resourceSpans_scopeSpans_spans_attributes_gen_ai_usage_completion_tokens": None,
                "resourceSpans_scopeSpans_spans_attributes_llm_usage_total_tokens": None,
                "tag": None,
                "model": None,
            }
        )
    return trace
