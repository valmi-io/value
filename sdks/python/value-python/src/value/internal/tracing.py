"""OpenTelemetry tracing initialization."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from .span_processor import UserContextSpanProcessor


def initialize_tracing(
    endpoint: str,
    service_name: str = "value-control-agent",
    console_export: bool = False,
    attributes: dict = None,
) -> trace.Tracer:
    """
    Initialize the OpenTelemetry tracer provider, processor, and exporter.

    Args:
        endpoint: OTLP endpoint for trace export
        service_name: Name of the service for resource attribution
        console_export: Enable console exporter for debugging

    Returns:
        Configured OpenTelemetry tracer
    """
    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
            "value.client.sdk": "value-python",
            **attributes,
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add user context span processor (must be first to run on all spans)
    user_context_processor = UserContextSpanProcessor()
    provider.add_span_processor(user_context_processor)

    # Create and add OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    otlp_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(otlp_processor)

    # Optionally add console exporter for debugging
    if console_export:
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        provider.add_span_processor(console_processor)

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Return tracer instance
    return trace.get_tracer(service_name)
