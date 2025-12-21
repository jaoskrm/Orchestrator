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
    
    # Read the generated code if it exists
    generated_code = ""
    code_file = task_dir / "main.py"
    if code_file.exists():
        generated_code = code_file.read_text(encoding="utf-8")
    else:
        generated_code = "# No code file generated"
    
    # Read pytest output
    pytest_events = [
        e for e in trace.get("events", [])
        if e.get("kind") == "pytest"
    ]

    if pytest_events:
        last_pytest = pytest_events[-1]["payload"]
        pytest_output = (
            (last_pytest.get("stdout_head") or "")
            + (last_pytest.get("stdout_tail") or "")
            + (last_pytest.get("stderr") or "")
        ).strip()
    else:
        pytest_output = "No pytest output recorded"

    
    # Extract workflow-specific metadata
    workflow = trace.get("workflow", "unknown")
    
    # Build status summary
    if result["passed"]:
        status = "✅ PASSED"
        status_color = "green"
    else:
        status = "❌ FAILED"
        status_color = "red"
    
    # Extract failure info from trace events (Phase A feature)
    failure_summary = ""
    rounds_info = ""
    full_regens = 0
    impl_repairs = 0
    
    for event in trace.get("events", []):
        if event.get("kind") == "failure_classification":
            failure_class = event.get("payload", {}).get("failure_class", "")
            failure_desc = event.get("payload", {}).get("summary", "")
            failure_summary = f"**{failure_class}**: {failure_desc}"
        
        if event.get("kind") == "round_start":
            payload = event.get("payload", {})
            full_regens = payload.get("full_regens_used", 0)
            impl_repairs = payload.get("impl_repairs_used", 0)
    
    rounds_info = f"Full Regens: {full_regens} | Impl Repairs: {impl_repairs}"
    
    # For coding workflow, use different "final answer"
    if workflow == "coding":
        final_answer = f"✅ Tests passed" if result["passed"] else failure_summary
    else:
        final_answer = trace.get("final_answer", "")
    
    return (
        status,
        rounds_info,
        failure_summary if failure_summary else "No failures (passed on first try)",
        generated_code,
        pytest_output,
        final_answer,
        json.dumps(trace, indent=2),
    )


def load_trace_history():
    """Load recent trace files for the history tab."""
    traces = sorted(TRACE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for p in traces[:30]:
        try:
            trace_data = json.loads(p.read_text(encoding="utf-8"))
            status = "✅" if trace_data.get("success", False) else "❌"
            workflow = trace_data.get("workflow", "unknown")
            task_id = trace_data.get("task_id", "unknown")
            
            # Count rounds
            rounds = len([e for e in trace_data.get("events", []) if e.get("kind") == "round_start"])
            
            rows.append([status, task_id, workflow, f"{rounds} rounds", p.name])
        except Exception:
            rows.append(["⚠️", "error", "error", "0", p.name])
    return rows


def view_trace_details(evt: gr.SelectData, trace_table):
    """Load full trace details when a row is clicked."""
    if not evt.index:
        return "", ""
    
    row = trace_table[evt.index[0]]
    trace_filename = row[-1]  # Last column is filename
    trace_path = TRACE_DIR / trace_filename
    
    if not trace_path.exists():
        return "Trace file not found", ""
    
    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    
    # Try to load the associated code
    task_id = trace_data.get("task_id", "")
    code_file = Path("runs") / task_id / "main.py"
    code = ""
    if code_file.exists():
        code = code_file.read_text(encoding="utf-8")
    else:
        code = "# Code file not found"
    
    return json.dumps(trace_data, indent=2), code


with gr.Blocks(title="Hybrid Orchestrator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Hybrid AI Orchestrator")
    gr.Markdown("**Phase A**: Test-freeze + implementation repair loop with failure classification")

    with gr.Tab("🏃 Run Task"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Task Configuration")
                task_name = gr.Textbox(
                    label="Task ID", 
                    value="demo_001",
                    placeholder="unique_task_identifier"
                )
                workflow = gr.Dropdown(
                    choices=["auto", "coding", "rag_qa", "reasoning", "science"],
                    value="auto",
                    label="Workflow Override",
                    info="Leave as 'auto' for automatic routing"
                )
                prompt = gr.Textbox(
                    label="Prompt", 
                    lines=10,
                    placeholder="Enter your task description here..."
                )
                context = gr.Files(
                    label="Context Docs (for RAG)", 
                    file_count="multiple",
                    file_types=[".txt", ".pdf", ".md", ".py"]
                )
                submit = gr.Button("▶️ Run Task", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Execution Summary")
                status = gr.Textbox(label="Status", scale=1)
                rounds_info = gr.Textbox(label="Repair Strategy", scale=1)
                failure_info = gr.Textbox(label="Failure Classification", lines=2)
        
        gr.Markdown("---")
        gr.Markdown("## 📝 Primary Artifacts")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Generated Code (main.py)")
                code_output = gr.Code(
                    language="python",
                    label="Implementation + Tests",
                    lines=25
                )
            
            with gr.Column():
                gr.Markdown("### 🧪 Pytest Results")
                pytest_output = gr.Code(
                    language="shell",
                    label="Test Execution Output",
                    lines=25
                )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                final_answer = gr.Textbox(label="Summary", lines=3)
            with gr.Column():
                pass  # Spacer
        
        with gr.Accordion("📊 Full Trace JSON (Debug)", open=False):
            trace_view = gr.Code(language="json", label="Complete Execution Trace")

        submit.click(
            fn=run_interactive,
            inputs=[task_name, prompt, workflow, context],
            outputs=[status, rounds_info, failure_info, code_output, pytest_output, final_answer, trace_view],
        )

    with gr.Tab("📜 History"):
        gr.Markdown("### Recent Task Executions")
        gr.Markdown("Click on any row to view full details")
        
        trace_table = gr.Dataframe(
            value=load_trace_history,
            headers=["Status", "Task ID", "Workflow", "Rounds", "Trace File"],
            interactive=False,
            wrap=True
        )
        
        refresh_btn = gr.Button("🔄 Refresh History")
        refresh_btn.click(fn=load_trace_history, outputs=trace_table)
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                history_trace = gr.Code(language="json", label="Trace Details", lines=20)
            with gr.Column():
                history_code = gr.Code(language="python", label="Generated Code", lines=20)
        
        trace_table.select(
            fn=view_trace_details,
            inputs=[trace_table],
            outputs=[history_trace, history_code]
        )

    with gr.Tab("📖 Documentation"):
        gr.Markdown("""
        ## Workflow Types
        
        ### 🔧 Coding
        - Generates Python code with pytest tests
        - **Phase A Features**:
          - Tests frozen after first valid generation
          - Implementation-only repairs for non-FORMAT failures
          - Failure classification: FORMAT | CONTRACT | RUNTIME_SAFETY | LOGIC
          - Format locks: AST parsing + markdown detection
          - Escalation: Auto-retry with full regen after 3 impl repairs
        
        ### 🧮 Reasoning
        - Step-by-step logical problem solving
        - Optional debate mode with critic
        
        ### 🔬 Science
        - Physics/chemistry/math problems
        - Handles units and numerical computation
        
        ### 📚 RAG QA
        - Question answering over documents
        - Retrieval + synthesis workflow
        
        ---
        
        ## Phase A Architecture
        
        ### Failure Classification
        
        | Class | Description | Action |
        |-------|-------------|--------|
        | **FORMAT** | Syntax errors, import failures | Full regeneration |
        | **CONTRACT** | Type errors, missing args | Implementation repair |
        | **RUNTIME_SAFETY** | AttributeError, NoneType access | Implementation repair |
        | **LOGIC** | Wrong computation, failed assertions | Implementation repair |
        
        ### Repair Loop Flow
        
        ```
        Round 1: Full Generation (impl + tests)
           ↓
        Tests frozen ✓
           ↓
        Round 2+: Implementation-only repairs
           ↓
        If FORMAT failure → Full regen
        If 3+ impl repairs fail → Escalate to full regen
        ```
        
        ### Success Indicators
        
        - ✅ Single-round pass (10-15 seconds)
        - ✅ Impl repairs < 3 before success
        - ✅ No FORMAT failures after round 1
        - ⚠️ Escalation to full regen means repair loop struggling
        
        ---
        
        ## Tips for Testing Phase A
        
        ### Easy Tasks (should pass in 1 round)
        - Simple classes with basic methods
        - Straightforward algorithms (factorial, fibonacci)
        - Data structures (stack, queue)
        
        ### Medium Tasks (triggers repair loop)
        - Classes with validation logic
        - Edge case handling
        - Optional parameter handling
        
        ### Hard Tasks (good for stress testing)
        - Complex state management
        - Multiple interacting classes
        - File I/O with error handling
        
        ---
        
        ## Next: Phase B Features (Planned)
        
        - 🎯 Function-level patching (text-diff based)
        - 🔍 Traceback line localization
        - 🧬 Guided constraint inference
        - 📊 Repair success metrics
        """)

demo.launch(share=False)