from pathlib import Path
from typing import Any, Dict, List, Tuple

from orchestrator.llm_clients import OllamaClient
from orchestrator.storage.traces import TraceWriter

def _call(worker: Dict[str, Any], prompt: str, temperature: float = 0.2) -> str:
    if worker["provider"] == "ollama":
        return OllamaClient().generate(model=worker["model"], prompt=prompt, temperature=temperature)
    raise ValueError(f"Unsupported provider: {worker['provider']}")

def _load_context(task_id: str) -> List[Tuple[str, str]]:
    ctx_dir = Path("runs") / task_id / "context"
    if not ctx_dir.exists():
        return []
    docs = []
    for p in sorted(ctx_dir.glob("*")):
        if p.is_file():
            docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
    return docs

def _simple_chunk(text: str, chunk_size: int = 1200) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def run_rag_qa_workflow(task_id: str, user_prompt: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    tracer = TraceWriter(task_id, "rag_qa", user_prompt, decision)

    workers = decision.get("workers", [])
    retriever = next((w for w in workers if w.get("role") == "retriever"), None)
    solver = next((w for w in workers if w.get("role") == "solver"), None)
    verifier = next((w for w in workers if w.get("role") in ("verifier", "critic")), None)
    if not solver:
        raise ValueError("rag_qa workflow requires a solver worker")

    docs = _load_context(task_id)
    if not docs:
        raise FileNotFoundError(f"No context/ docs found for {task_id}. Add runs/{task_id}/context/*")

    # Chunk docs (v0 naive)
    chunks = []
    for name, txt in docs:
        for i, ch in enumerate(_simple_chunk(txt)):
            chunks.append((f"{name}#chunk{i}", ch))

    # Retriever chooses top chunks (v0: LLM selects by reading titles + snippets)
    shortlist = chunks[:40]  # keep prompt bounded
    retrieve_prompt = (
        "Select the 5 most relevant chunks for the user question.\n"
        "Return ONLY a JSON list of chunk_ids.\n\n"
        f"Question:\n{user_prompt}\n\n"
        f"Chunks:\n" + "\n".join([f"- {cid}: {c[:200].replace('\\n',' ')}" for cid, c in shortlist]) + "\n\n"
        "JSON:"
    )
    if retriever:
        raw = _call(retriever, retrieve_prompt, temperature=0.1)
    else:
        raw = _call(solver, retrieve_prompt, temperature=0.1)
    tracer.log("retriever_raw", text=raw[:1500])

    # Parse chosen ids (best-effort)
    chosen_ids = []
    try:
        import json
        chosen_ids = json.loads(raw)
        if not isinstance(chosen_ids, list):
            chosen_ids = []
    except Exception:
        chosen_ids = []

    chosen = [c for c in chunks if c[0] in set(chosen_ids)]
    if not chosen:
        chosen = chunks[:5]

    context_block = "\n\n".join([f"[{cid}]\n{c}" for cid, c in chosen])
    tracer.log("chosen_chunks", ids=[cid for cid, _ in chosen])

    answer_prompt = (
        "Answer the question using ONLY the provided context.\n"
        "Cite chunk ids like [doc#chunkN] after relevant sentences.\n"
        "If not answerable from context, say 'Not enough information in provided documents.'\n\n"
        f"Question:\n{user_prompt}\n\n"
        f"Context:\n{context_block}\n"
    )
    draft = _call(solver, answer_prompt, temperature=0.2)
    tracer.log("solver_draft", model=solver["model"], text=draft[:2000])

    final = draft
    if verifier:
        verify_prompt = (
            "Check that every claim is supported by the cited context chunks.\n"
            "Remove/repair any unsupported claims and keep citations.\n\n"
            f"Question:\n{user_prompt}\n\n"
            f"Answer:\n{draft}\n\n"
            f"Context:\n{context_block}\n"
        )
        final = _call(verifier, verify_prompt, temperature=0.1)
        tracer.log("verifier_final", model=verifier["model"], text=final[:2000])

    tracer.set_result(success=True, final_answer=final)
    trace_path = tracer.flush()
    return {"task_id": task_id, "answer": final, "trace": str(trace_path)}
