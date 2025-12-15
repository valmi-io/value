"""
AI Pipeline Simulation - Demonstrates a complete document processing workflow.

Simulates: Object Detection → Classification → OCR → RAG → LLM
"""

import asyncio
import os
import random
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from value import initialize_async
from opentelemetry import trace

# Get agent secret from environment variable
AGENT_SECRET = os.getenv("VALUE_AGENT_SECRET", "your-agent-secret")

INVOICE_TYPES = ["invoice", "receipt", "bill", "purchase_order"]
VENDORS = ["Amazon", "Google", "Microsoft", "Apple", "Uber", "Lyft"]
CURRENCIES = ["USD", "EUR", "GBP"]


class PipelineSimulator:
    def __init__(self, client):
        self.client = client
        self.tracer = client.tracer

    async def run_pipeline(self):
        user_id = f"user_{random.randint(1000, 9999)}"
        anonymous_id = str(uuid.uuid4())
        invoice_id = f"INV-{random.randint(10000, 99999)}"

        print(f"\n--- Pipeline: {invoice_id} (User: {user_id}) ---")

        await self._object_detection(user_id, anonymous_id, invoice_id)
        await self._object_classification(user_id, anonymous_id, invoice_id)
        text = await self._ocr(user_id, anonymous_id, invoice_id)
        context = await self._rag(user_id, anonymous_id, invoice_id, text)
        await self._llm(user_id, anonymous_id, invoice_id, context)

    async def _delay(self):
        delay = random.uniform(1.0, 2.0)
        await asyncio.sleep(delay)
        return delay

    async def _object_detection(self, user_id: str, anonymous_id: str, invoice_id: str):
        print("1. Object Detection...")
        await self._delay()

        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="object_detection",
                **{
                    "value.action.description": f"Detected objects in {invoice_id}",
                    "invoice_id": invoice_id,
                    "model": "yolov8-invoice",
                    "confidence_score": random.uniform(0.85, 0.99),
                    "detected_boxes": random.randint(3, 10),
                },
            )

    async def _object_classification(self, user_id: str, anonymous_id: str, invoice_id: str):
        print("2. Classification...")
        await self._delay()

        doc_type = random.choice(INVOICE_TYPES)
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="object_classification",
                **{
                    "value.action.description": f"Classified as {doc_type}",
                    "invoice_id": invoice_id,
                    "document_type": doc_type,
                    "confidence": random.uniform(0.90, 0.99),
                },
            )

    async def _ocr(self, user_id: str, anonymous_id: str, invoice_id: str) -> str:
        print("3. OCR...")
        await self._delay()

        vendor = random.choice(VENDORS)
        total = round(random.uniform(10.0, 500.0), 2)
        currency = random.choice(CURRENCIES)
        text = f"Invoice from {vendor}\nTotal: {total} {currency}"

        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="ocr_processing",
                **{
                    "value.action.description": f"Extracted text from {invoice_id}",
                    "invoice_id": invoice_id,
                    "ocr_engine": "tesseract",
                    "text_length": len(text),
                },
            )
        return text

    async def _rag(self, user_id: str, anonymous_id: str, invoice_id: str, query: str) -> str:
        print("4. RAG Retrieval...")
        await self._delay()

        chunks = random.randint(2, 5)
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="rag_retrieval",
                **{
                    "value.action.description": f"Retrieved {chunks} chunks",
                    "invoice_id": invoice_id,
                    "vector_db": "chroma",
                    "chunks_count": chunks,
                },
            )
        return f"Context for {query}"

    async def _llm(self, user_id: str, anonymous_id: str, invoice_id: str, context: str):
        print("5. LLM Call...")
        delay = await self._delay()

        if random.choice([True, False]):
            await self._gemini_call(user_id, anonymous_id, invoice_id, delay)
        else:
            await self._ollama_call(user_id, anonymous_id, invoice_id, delay)

    async def _gemini_call(self, user_id: str, anonymous_id: str, invoice_id: str, duration: float):
        print("   -> Gemini API")
        model = "gemini-2.5-flash"

        start_time = time.time_ns()
        end_time = start_time + int(duration * 1e9)

        span = self.tracer.start_span(
            name="gemini.generate_content",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "gen_ai.system": "Google",
                "gen_ai.request.model": model,
                "gen_ai.response.model": model,
                "gen_ai.usage.prompt_tokens": str(random.randint(10, 50)),
                "gen_ai.usage.completion_tokens": str(random.randint(50, 200)),
                "value.action.user_id": user_id,
                "value.action.anonymous_id": anonymous_id,
            },
            start_time=start_time,
        )
        span.end(end_time=end_time)

        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="process_gemini_response",
                **{
                    "value.action.description": f"Processed {model} response",
                    "model": model,
                    "prompt_type": "summarization",
                },
            )

    async def _ollama_call(self, user_id: str, anonymous_id: str, invoice_id: str, duration: float):
        print("   -> Ollama API")
        model = "llama3"

        start_time = time.time_ns()
        end_time = start_time + int(duration * 1e9)

        span = self.tracer.start_span(
            name="ollama.generate",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "value.action.llm.model": model,
                "value.action.llm.input_tokens": str(random.randint(10, 50)),
                "value.action.llm.output_tokens": str(random.randint(50, 200)),
            },
            start_time=start_time,
        )
        span.end(end_time=end_time)

        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="process_ollama_response",
                **{
                    "value.action.description": f"Processed {model} response",
                    "model": model,
                    "prompt_type": "extraction",
                },
            )


async def main():
    print("Initializing Value SDK...")
    client = await initialize_async(agent_secret=AGENT_SECRET)
    simulator = PipelineSimulator(client)

    print("Starting simulation (Ctrl+C to stop)...")
    try:
        while True:
            await simulator.run_pipeline()
            print("Waiting 5 seconds...")
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())
