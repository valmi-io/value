#!/usr/bin/env python3
"""Seed demo data for the Value platform."""

import asyncio
import random
import time
import uuid
from datetime import datetime, timezone

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .utils.config import SEED_EVENTS_PER_DAY, SEED_MONTHS
from .utils.control_plane_seeder import ControlPlaneSeeder
from .utils.clickhouse_seeder import ClickHouseSeeder
from .utils.seeded_data import SeededDataManager

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()
app = typer.Typer(help="Seed demo data for Value platform")

INVOICE_TYPES = ["invoice", "receipt", "bill", "purchase_order"]
VENDORS = ["Amazon", "Google", "Microsoft", "Apple", "Uber", "Lyft", "Walmart", "Target"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD"]


def display_summary(title: str, data: dict):
    table = Table(title=title)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(str(key), str(value))
    console.print(table)


async def send_sample_trace(agent_secret: str, agent_name: str):
    from value import AsyncValueClient

    try:
        client = AsyncValueClient(secret=agent_secret)
        await client.initialize()
        anonymous_id = str(uuid.uuid4())
        user_id = f"user_{anonymous_id[:8]}"
        with client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="seed_init_trace",
                **{"value.action.description": "Initial trace to create ClickHouse tables"},
            )
        await asyncio.sleep(1)
        console.print(f"  [green]Sent sample trace via agent: {agent_name}[/green]")
        return True
    except Exception as e:
        console.print(f"  [red]Failed to send sample trace: {e}[/red]")
        return False


def wait_for_clickhouse_table(ch_seeder: ClickHouseSeeder, timeout: int = 60, interval: int = 2) -> bool:
    elapsed = 0
    while elapsed < timeout:
        if ch_seeder.table_exists():
            return True
        time.sleep(interval)
        elapsed += interval
        console.print(f"  [dim]Waiting for ClickHouse table... ({elapsed}s)[/dim]")
    return False


@app.command()
def seed_past(
    months: int = typer.Option(SEED_MONTHS, "--months", "-m", help="Number of months of historical data to generate"),
    events_per_day: int = typer.Option(
        SEED_EVENTS_PER_DAY, "--events-per-day", "-e", help="Base number of events per day"
    ),
    skip_control_plane: bool = typer.Option(False, "--skip-control-plane", help="Skip seeding Control Plane entities"),
    skip_clickhouse: bool = typer.Option(False, "--skip-clickhouse", help="Skip seeding ClickHouse data"),
    no_clear: bool = typer.Option(False, "--no-clear", help="Don't clear existing ClickHouse data"),
):
    """Seed historical data for the past N months."""
    console.print(f"\n[bold blue]Seeding {months} months of historical data[/bold blue]")
    console.print(f"Base events per day: {events_per_day}\n")

    agents_data = []

    if not skip_control_plane:
        console.print("[bold]Phase 1: Seeding Control Plane entities...[/bold]")
        cp_seeder = ControlPlaneSeeder(seed_months=months)
        try:
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("Creating entities...", total=None)
                summary = cp_seeder.seed_all()
                progress.update(task, description="Done!")
            display_summary("Control Plane Summary", summary)
            agents_data = cp_seeder.get_agent_data_for_traces()
            if cp_seeder.save_seeded_data():
                console.print("[green]Saved agent data to seeded_agents.json[/green]")
            else:
                console.print("[yellow]Warning: Failed to save agent data[/yellow]")

            if agents_data and not skip_clickhouse:
                console.print("\n[bold]Phase 1.5: Sending sample trace to initialize ClickHouse tables...[/bold]")
                first_agent_id = agents_data[0].get("agent_id")
                first_agent_name = agents_data[0].get("agent_name")
                agent_secrets = cp_seeder.refresh_agent_secrets([first_agent_id])
                if agent_secrets:
                    first_secret = list(agent_secrets.values())[0]
                    asyncio.run(send_sample_trace(first_secret, first_agent_name))
                else:
                    console.print("[yellow]Warning: Could not get agent secret for sample trace[/yellow]")
        except Exception as e:
            console.print(f"[red]Control Plane seeding failed: {e}[/red]")
            logger.exception("Control Plane seeding error")
        finally:
            cp_seeder.close()
    else:
        console.print("[yellow]Skipping Control Plane seeding[/yellow]")
        data_manager = SeededDataManager()
        if data_manager.load():
            agents_data = data_manager.get_agents()
            console.print(f"[green]Loaded {len(agents_data)} agents from seeded_agents.json[/green]")

    if not skip_clickhouse:
        console.print("\n[bold]Phase 2: Seeding ClickHouse trace data...[/bold]")
        ch_seeder = ClickHouseSeeder()
        try:
            ch_seeder.connect()

            if not skip_control_plane:
                console.print("  [dim]Waiting for ClickHouse table to be created by OTEL collector...[/dim]")
                if wait_for_clickhouse_table(ch_seeder, timeout=300, interval=2):
                    console.print("  [green]ClickHouse table is ready![/green]")
                else:
                    console.print("  [yellow]Table not found after timeout, will create it manually[/yellow]")

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("Generating traces...", total=None)
                summary = ch_seeder.seed_past_data(
                    agents_data=agents_data, months=months, events_per_day=events_per_day, clear_existing=not no_clear
                )
                progress.update(task, description="Done!")
            display_summary("ClickHouse Summary", summary)
            row_count = ch_seeder.get_row_count()
            console.print(f"\n[green]Total rows in measure table: {row_count:,}[/green]")

            console.print("\n[bold]Phase 3: Seeding live_metrics table...[/bold]")
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
            ) as progress:
                task = progress.add_task("Aggregating metrics...", total=None)
                live_metrics_summary = ch_seeder.seed_live_metrics()
                progress.update(task, description="Done!")
            display_summary("Live Metrics Summary", live_metrics_summary)
            live_metrics_count = ch_seeder.get_live_metrics_count()
            console.print(f"\n[green]Total rows in live_metrics table: {live_metrics_count:,}[/green]")
        except Exception as e:
            console.print(f"[red]ClickHouse seeding failed: {e}[/red]")
            logger.exception("ClickHouse seeding error")
        finally:
            ch_seeder.close()
    else:
        console.print("[yellow]Skipping ClickHouse seeding[/yellow]")

    console.print("\n[bold green]Historical data seeding complete![/bold green]")
    console.print("You can now run [cyan]seed-live[/cyan] to continue with real-time data.")


class PipelineSimulator:
    def __init__(self, client, anonymous_ids: list):
        self.client = client
        self.tracer = client.tracer
        self.console = console
        self.anonymous_ids = anonymous_ids

    async def simulate_delay(self):
        delay = random.uniform(0.5, 1.5)
        await asyncio.sleep(delay)
        return delay

    async def run_pipeline(self):
        user_id = f"user_{random.randint(1000, 9999)}"
        invoice_id = f"INV-{random.randint(10000, 99999)}"
        anonymous_id = self.anonymous_ids[0] if self.anonymous_ids else str(uuid.uuid4())
        self.console.print(
            f"  [dim]Pipeline for {invoice_id} (User: {user_id}, Anonymous IDs: {len(self.anonymous_ids)})[/dim]"
        )
        await self._step_object_detection(user_id, anonymous_id, invoice_id)
        await self._step_object_classification(user_id, anonymous_id, invoice_id)
        detected_text = await self._step_ocr(user_id, anonymous_id, invoice_id)
        context = await self._step_rag(user_id, anonymous_id, invoice_id, detected_text)
        await self._step_llm(user_id, anonymous_id, invoice_id, context)

    async def _step_object_detection(self, user_id: str, anonymous_id: str, invoice_id: str):
        await self.simulate_delay()
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="object_detection",
                **{
                    "value.action.description": f"Detected {random.randint(3, 10)} objects in {invoice_id}",
                    "invoice_id": invoice_id,
                    "model": "yolov8-invoice",
                    "confidence_score": random.uniform(0.85, 0.99),
                    "detected_boxes": random.randint(3, 10),
                    "processing_device": "cuda:0",
                },
            )

    async def _step_object_classification(self, user_id: str, anonymous_id: str, invoice_id: str):
        await self.simulate_delay()
        doc_type = random.choice(INVOICE_TYPES)
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="object_classification",
                **{
                    "value.action.description": f"Classified document as {doc_type}",
                    "invoice_id": invoice_id,
                    "document_type": doc_type,
                    "classification_confidence": random.uniform(0.90, 0.99),
                    "model": "resnet50-finetuned",
                },
            )

    async def _step_ocr(self, user_id: str, anonymous_id: str, invoice_id: str) -> str:
        await self.simulate_delay()
        vendor = random.choice(VENDORS)
        total = round(random.uniform(10.0, 500.0), 2)
        currency = random.choice(CURRENCIES)
        date = datetime.now().strftime("%Y-%m-%d")
        detected_text = f"Invoice from {vendor}\nDate: {date}\nTotal: {total} {currency}"
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="ocr_processing",
                **{
                    "value.action.description": f"Extracted text from {invoice_id}",
                    "invoice_id": invoice_id,
                    "ocr_engine": "tesseract",
                    "language": "eng",
                    "text_length": len(detected_text),
                    "confidence": random.uniform(0.8, 0.95),
                },
            )
        return detected_text

    async def _step_rag(self, user_id: str, anonymous_id: str, invoice_id: str, query_text: str) -> str:
        await self.simulate_delay()
        chunks_retrieved = random.randint(2, 5)
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="rag_retrieval",
                **{
                    "value.action.description": f"Retrieved {chunks_retrieved} chunks for context",
                    "invoice_id": invoice_id,
                    "vector_db": "chroma",
                    "embedding_model": "text-embedding-3-small",
                    "chunks_count": chunks_retrieved,
                    "top_k": 5,
                },
            )
        return f"Context for {query_text}"

    async def _step_llm(self, user_id: str, anonymous_id: str, invoice_id: str, context: str):
        delay = await self.simulate_delay()
        if random.choice([True, False]):
            await self._simulate_gemini_call(user_id, anonymous_id, invoice_id, delay)
        else:
            await self._simulate_ollama_call(user_id, anonymous_id, invoice_id, delay)

    async def _simulate_gemini_call(self, user_id: str, anonymous_id: str, invoice_id: str, duration: float):
        from opentelemetry import trace

        model = "gemini-2.5-flash"
        prompt = f"Summarize invoice {invoice_id}"
        response_text = f"This is a summary of invoice {invoice_id} from Gemini."
        tracer = self.client.tracer
        start_time = time.time_ns()
        end_time = start_time + int(duration * 1e9)
        attributes = {
            "gen_ai.system": "Google",
            "llm.request.type": "completion",
            "value.action.user_id": user_id,
            "value.action.anonymous_id": anonymous_id,
            "gen_ai.prompt.0.content": f'[{{"type": "text", "text": "{prompt}"}}]',
            "gen_ai.prompt.0.role": "user",
            "gen_ai.request.model": model,
            "gen_ai.completion.0.content": response_text,
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.response.model": model,
            "llm.usage.total_tokens": random.randint(100, 1000),
            "gen_ai.usage.completion_tokens": random.randint(50, 200),
            "gen_ai.usage.prompt_tokens": random.randint(10, 50),
        }
        span = tracer.start_span(
            name="gemini.generate_content", kind=trace.SpanKind.CLIENT, attributes=attributes, start_time=start_time
        )
        span.end(end_time=end_time)
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="process_gemini_response",
                **{
                    "value.action.description": f"Received response from {model}",
                    "custom.model": model,
                    "custom.response_length": len(response_text),
                    "custom.prompt_type": "summarization",
                },
            )

    async def _simulate_ollama_call(self, user_id: str, anonymous_id: str, invoice_id: str, duration: float):
        from opentelemetry import trace

        model = "llama3"
        prompt = f"Extract details from {invoice_id}"
        response_text = f"Extracted details for {invoice_id} using Ollama."
        tracer = self.client.tracer
        start_time = time.time_ns()
        end_time = start_time + int(duration * 1e9)
        attributes = {
            "value.action.llm.model": model,
            "value.action.llm.input_tokens": random.randint(10, 50),
            "value.action.llm.output_tokens": random.randint(50, 200),
            "value.action.llm.total_tokens": random.randint(100, 1000),
            "value.action.llm.prompt": prompt,
            "value.action.llm.response": response_text,
        }
        span = tracer.start_span(
            name="ollama.generate", kind=trace.SpanKind.CLIENT, attributes=attributes, start_time=start_time
        )
        span.end(end_time=end_time)
        with self.client.action_context(anonymous_id=anonymous_id, user_id=user_id) as ctx:
            ctx.send(
                action_name="process_ollama_response",
                **{
                    "value.action.description": f"Received response from {model}",
                    "custom.model": model,
                    "custom.response_length": len(response_text),
                    "custom.prompt_type": "extraction",
                },
            )


async def run_live_simulation(
    data_manager: SeededDataManager, agent_secrets: dict, interval: float, pipelines_per_interval: int
):
    from value import AsyncValueClient

    agents = data_manager.get_agents()
    if not agents:
        console.print("[red]No agents found in seeded data[/red]")
        return
    console.print("[bold]Initializing Value SDK clients for each agent...[/bold]")
    clients = {}
    for agent in agents:
        agent_id = agent.get("agent_id")
        secret = agent_secrets.get(agent_id)
        if not secret:
            console.print(f"[yellow]No secret for agent {agent_id}, skipping[/yellow]")
            continue
        try:
            client = AsyncValueClient(secret=secret)
            await client.initialize()
            clients[agent_id] = client
            console.print(f"  [green]Initialized client for {agent.get('agent_name')}[/green]")
        except Exception as e:
            console.print(f"  [red]Failed to initialize client for {agent_id}: {e}[/red]")
    if not clients:
        console.print("[red]No SDK clients initialized[/red]")
        return
    console.print(f"\n[green]Initialized {len(clients)} SDK clients[/green]\n")
    iteration = 0
    total_pipelines = 0
    try:
        while True:
            iteration += 1
            timestamp = datetime.now(timezone.utc)
            console.print(
                f"[dim][{timestamp.strftime('%H:%M:%S')}][/dim] [bold]Iteration {iteration}[/bold] - Running {pipelines_per_interval} pipeline(s)..."
            )
            for _ in range(pipelines_per_interval):
                agent = random.choice(agents)
                agent_id = agent.get("agent_id")
                client = clients.get(agent_id)
                if not client:
                    continue
                anonymous_ids = data_manager.get_next_anonymous_ids(agent_id)
                if not anonymous_ids:
                    anonymous_ids = [str(uuid.uuid4())]
                simulator = PipelineSimulator(client, anonymous_ids)
                await simulator.run_pipeline()
                total_pipelines += 1
            console.print(f"  [green]Total pipelines: {total_pipelines}[/green]\n")
            await asyncio.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping live simulation...[/yellow]")
    console.print(f"\n[green]Live simulation complete. Total pipelines: {total_pipelines}[/green]")


@app.command()
def seed_live(
    interval: float = typer.Option(5.0, "--interval", "-i", help="Seconds between pipeline iterations"),
    pipelines_per_interval: int = typer.Option(1, "--pipelines", "-p", help="Number of pipelines to run per interval"),
):
    """Seed live data in real-time using the Value SDK."""
    console.print("\n[bold blue]Starting live data simulation (using Value SDK)[/bold blue]")
    console.print(f"Interval: {interval}s, Pipelines per interval: {pipelines_per_interval}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    console.print("[bold]Loading seeded agent data...[/bold]")
    data_manager = SeededDataManager()
    if not data_manager.exists():
        console.print("[red]Error: seeded_agents.json not found[/red]")
        console.print("Run 'seed-past' first to create agent data")
        raise typer.Exit(1)
    if not data_manager.load():
        console.print("[red]Error: Failed to load seeded_agents.json[/red]")
        raise typer.Exit(1)
    agents = data_manager.get_agents()
    console.print(f"[green]Loaded {len(agents)} agents[/green]")

    console.print("\n[bold]Refreshing agent secrets...[/bold]")
    cp_seeder = ControlPlaneSeeder()
    try:
        agent_ids = [agent.get("agent_id") for agent in agents]
        agent_secrets = cp_seeder.refresh_agent_secrets(agent_ids)
        console.print(f"[green]Refreshed secrets for {len(agent_secrets)} agents[/green]\n")
    except Exception as e:
        console.print(f"[red]Error refreshing secrets: {e}[/red]")
        raise typer.Exit(1)
    finally:
        cp_seeder.close()
    if not agent_secrets:
        console.print("[red]Error: No agent secrets obtained[/red]")
        raise typer.Exit(1)

    try:
        asyncio.run(run_live_simulation(data_manager, agent_secrets, interval, pipelines_per_interval))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Live simulation error")
        raise typer.Exit(1)


@app.command()
def status():
    """Check the current status of seeded data."""
    console.print("\n[bold]Checking data status...[/bold]\n")

    console.print("[bold]ClickHouse Status:[/bold]")
    ch_seeder = ClickHouseSeeder()
    try:
        ch_seeder.connect()
        row_count = ch_seeder.get_row_count()
        result = ch_seeder.client.query(
            "SELECT min(created_at), max(created_at) FROM scratch_traces_flatten_measure_table"
        )
        if result.result_rows and result.result_rows[0][0]:
            min_date, max_date = result.result_rows[0]
            console.print(f"  Measure Table Rows: {row_count:,}")
            console.print(f"  Date range: {min_date} to {max_date}")
        else:
            console.print(f"  Measure Table Rows: {row_count:,}")
            console.print("  No data found")
        live_metrics_count = ch_seeder.get_live_metrics_count()
        console.print(f"\n  Live Metrics Table Rows: {live_metrics_count:,}")
        try:
            lm_result = ch_seeder.client.query("SELECT min(metric_date), max(metric_date) FROM live_metrics")
            if lm_result.result_rows and lm_result.result_rows[0][0]:
                lm_min_date, lm_max_date = lm_result.result_rows[0]
                console.print(f"  Live Metrics Date range: {lm_min_date} to {lm_max_date}")
        except Exception:
            console.print("  Live Metrics: No data found")
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
    finally:
        ch_seeder.close()

    console.print("\n[bold]Control Plane Status:[/bold]")
    cp_seeder = ControlPlaneSeeder()
    try:
        cp_seeder.get_current_workspace()
        console.print(f"  Workspace: {cp_seeder.workspace_id}")
        console.print(f"  Organization: {cp_seeder.organization_id}")
        agents = cp_seeder._make_request("GET", "/api/v1/agents")
        console.print(f"  Agents: {len(agents.get('agents', []))}")
        accounts = cp_seeder._make_request("GET", "/api/v1/accounts")
        console.print(f"  Accounts: {len(accounts.get('accounts', []))}")
    except Exception as e:
        console.print(f"  [red]Error: {e}[/red]")
    finally:
        cp_seeder.close()


if __name__ == "__main__":
    app()
