# orchestrator/cli.py
import json
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from orchestrator.app import main
from orchestrator.storage.traces import TRACE_DIR

app = typer.Typer()
console = Console()

@app.command()
def run(task_id: str, workflow: str = typer.Option(None, help="Override workflow: coding|rag_qa|reasoning|science")):
    """Run orchestrator"""
    with Progress() as progress:
        t = progress.add_task(f"[cyan]Running {task_id}...", total=100)
        result = main(task_id, workflow_override=workflow)
        progress.update(t, completed=100)

    table = Table(title=f"Results: {task_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Status", "PASS" if result["passed"] else "FAIL")
    table.add_row("Score", f"{result['score']:.2f}")
    table.add_row("Workflow", result["workflow"])
    console.print(table)
    console.print(Panel(result["reason"], title="Judge Verdict"))

@app.command()
def traces(n: int = 10):
    """List recent traces"""
    traces = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]
    table = Table(title="Recent Traces")
    table.add_column("Run ID")
    table.add_column("Task")
    table.add_column("Workflow")
    table.add_column("Success")

    for t in traces:
        data = json.loads(t.read_text(encoding="utf-8"))
        table.add_row(data["run_id"], data["task_id"], data["workflow"], "✓" if data.get("success") else "✗")

    console.print(table)

if __name__ == "__main__":
    app()
