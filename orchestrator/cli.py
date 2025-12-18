from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress
import typer

app = typer.Typer()
console = Console()

@app.command()
def run(task_id: str, watch: bool = False):
    """Run orchestrator with live progress"""
    with Progress() as progress:
        task = progress.add_task(f"[cyan]Running {task_id}...", total=100)
        
        # Hook into tracer events
        result = main(task_id)
        
        progress.update(task, completed=100)
    
    # Pretty results
    table = Table(title=f"Results: {task_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Status", "✅ PASS" if result["passed"] else "❌ FAIL")
    table.add_row("Score", f"{result['score']:.2f}")
    table.add_row("Workflow", result["workflow"])
    
    console.print(table)
    console.print(Panel(result["reason"], title="Judge Verdict"))

@app.command()
def traces():
    """List recent traces"""
    traces = sorted(TRACE_DIR.glob("*.json"), reverse=True)[:10]
    table = Table(title="Recent Traces")
    table.add_column("Task")
    table.add_column("Workflow")
    table.add_column("Score")
    
    for t in traces:
        data = json.loads(t.read_text())
        table.add_row(data["task_id"], data["workflow"], "✓" if data.get("success") else "✗")
    
    console.print(table)

if __name__ == "__main__":
    app()
