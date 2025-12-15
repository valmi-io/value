"""Basic OpenTelemetry tracing example (without Value SDK)."""

import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

resource = Resource(attributes={"service.name": "otel-example"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

tracer = trace.get_tracer(__name__)


def main():
    with tracer.start_as_current_span("parent-operation") as parent:
        parent.set_attribute("user.id", "123")
        parent.set_attribute("operation.type", "example")
        print("Parent span created")

        with tracer.start_as_current_span("child-operation") as child:
            child.set_attribute("step", "processing")
            print("Child span created")
            time.sleep(0.1)

        with tracer.start_as_current_span("another-child") as child:
            child.set_attribute("step", "finalizing")
            print("Another child span created")
            time.sleep(0.05)

    print("Traces sent!")
    tracer_provider.force_flush()


if __name__ == "__main__":
    main()
