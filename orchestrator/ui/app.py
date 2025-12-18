# orchestrator/ui/app.py
import json
import shutil
from pathlib import Path

import gradio as gr

from orchestrator.app import main as run_task
from orchestrator.storage.traces import TRACE_DIR


def run_interactive(task_name, prompt_text, workflow_override, context_files):
    task_dir = Path("runs") / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")

    if context_files:
        ctx_dir = task_dir / "context"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        for f in context_files:
            src = Path(f.name)
            dst = ctx_dir / src.name
            shutil.copyfile(src, dst)

    override = None if workflow_override in (None, "", "auto") else workflow_override
    result = run_task(task_name, workflow_override=override)

    trace = json.loads(Path(result["trace"]).read_text(encoding="utf-8"))
    return (
        ("PASSED" if result["passed"] else "FAILED"),
        f"{result['score']:.2f}",
        result["reason"],
        trace.get("final_answer", ""),
        json.dumps(trace, indent=2),
    )


with gr.Blocks(title="Hybrid Orchestrator") as demo:
    gr.Markdown("# Hybrid AI Orchestrator")

    with gr.Row():
        with gr.Column():
            task_name = gr.Textbox(label="Task ID", value="demo_001")
            prompt = gr.Textbox(label="Prompt", lines=6)
            workflow = gr.Dropdown(
                choices=["auto", "coding", "rag_qa", "reasoning", "science"],
                value="auto",
                label="Workflow Override",
            )
            context = gr.Files(label="Context Docs (for RAG)", file_count="multiple")
            submit = gr.Button("Run", variant="primary")

        with gr.Column():
            status = gr.Textbox(label="Status")
            score = gr.Textbox(label="Score")
            reason = gr.Textbox(label="Judge Reason")
            answer = gr.Textbox(label="Final Answer", lines=8)

    trace_view = gr.Code(language="json", label="Trace JSON")

    submit.click(
        fn=run_interactive,
        inputs=[task_name, prompt, workflow, context],
        outputs=[status, score, reason, answer, trace_view],
    )

    with gr.Tab("History"):
        def load_traces():
            traces = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            rows = []
            for p in traces[:30]:
                rows.append([p.name, p.read_text(encoding="utf-8")[:200]])
            return rows

        gr.Dataframe(value=load_traces, headers=["Trace file", "Preview"], interactive=False)

demo.launch(share=False)
