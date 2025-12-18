import gradio as gr
import json
from pathlib import Path
from orchestrator.app import main as run_task
from orchestrator.storage.traces import TRACE_DIR

def run_interactive(task_name, prompt_text, workflow_override, context_files):
    # Auto-create task dir
    task_dir = Path("runs") / task_name
    task_dir.mkdir(exist_ok=True)
    
    # Write prompt
    (task_dir / "prompt.txt").write_text(prompt_text)
    
    # Handle context uploads for RAG
    if context_files:
        ctx_dir = task_dir / "context"
        ctx_dir.mkdir(exist_ok=True)
        for f in context_files:
            Path(f.name).rename(ctx_dir / Path(f.name).name)
    
    # Run orchestrator
    result = run_task(task_name)
    
    # Load and format trace
    trace = json.loads(Path(result["trace"]).read_text())
    
    return {
        "status": "✅ PASSED" if result["passed"] else "❌ FAILED",
        "score": f"{result['score']:.2f}",
        "reason": result["reason"],
        "trace_json": json.dumps(trace, indent=2),
        "final_answer": trace.get("final_answer", "")
    }

# UI Layout
with gr.Blocks(title="Hybrid Orchestrator") as demo:
    gr.Markdown("# 🤖 Hybrid AI Orchestrator")
    
    with gr.Row():
        with gr.Column():
            task_name = gr.Textbox(label="Task ID", value="demo_001")
            prompt = gr.Textbox(label="Prompt", lines=5)
            workflow = gr.Dropdown(
                choices=["auto", "coding", "rag_qa", "reasoning", "science"],
                value="auto",
                label="Workflow Override"
            )
            context = gr.Files(label="Context Docs (for RAG)", file_count="multiple")
            submit = gr.Button("🚀 Run", variant="primary")
        
        with gr.Column():
            status = gr.Textbox(label="Status")
            score = gr.Textbox(label="Score")
            reason = gr.Textbox(label="Judge Reason")
            answer = gr.Textbox(label="Final Answer", lines=8)
    
    with gr.Accordion("📊 Trace JSON", open=False):
        trace_view = gr.Code(language="json", label="Full Trace")
    
    submit.click(
        fn=run_interactive,
        inputs=[task_name, prompt, workflow, context],
        outputs=[status, score, reason, trace_view, answer]
    )
    
    # Trace history viewer
    with gr.Tab("📂 History"):
        def load_traces():
            traces = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            return [[p.stem, p.read_text()[:200]] for p in traces[:20]]
        
        gr.Dataframe(value=load_traces, headers=["Task", "Preview"], interactive=False)

demo.launch(share=False)
