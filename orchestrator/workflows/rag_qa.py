from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from orchestrator.llm_clients import OllamaClient
from orchestrator.storage.traces import TraceWriter


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _call(worker: Dict[str, Any], prompt: str, temperature: float = 0.2) -> str:
    if worker["provider"] == "ollama":
        return OllamaClient().generate(model=worker["model"], prompt=prompt, temperature=temperature)
    raise ValueError(f"Unsupported provider: {worker['provider']}")


def _load_context(task_id: str) -> List[Tuple[str, str]]:
    ctx_dir = Path("runs") / task_id / "context"
    if not ctx_dir.exists():
        return []
    docs: List[Tuple[str, str]] = []
    for p in sorted(ctx_dir.glob("*")):
        if p.is_file():
            docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
    return docs


def _simple_chunk(text: str, chunk_size: int = 1200) -> List[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            # drop first (``````)
            return "\n".join(lines[1:-1]).strip()
    return s


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
        raise FileNotFoundError(f"No context docs found for {task_id}. Add runs/{task_id}/context/*")

    # Chunk docs (v0 naive)
    chunks: List[Tuple[str, str]] = []
    for name, txt in docs:
        for i, ch in enumerate(_simple_chunk(txt)):
            chunks.append((f"{name}#chunk{i}", ch))

    shortlist = chunks[:40]  # keep prompt bounded
    retrieve_prompt = (
        "Select the 5 most relevant chunks for the user question.\n"
        "Return ONLY a JSON list of chunk_ids.\n\n"
        f"Question:\n{user_prompt}\n\n"
        "Chunks:\n"
        + "\n".join([f"- {cid}: {c[:200].replace('\\n',' ')}" for cid, c in shortlist])
        + "\n\nJSON:"
    )

    raw_chunk_ids = _call(retriever or solver, retrieve_prompt, temperature=0.1)
    tracer.log("retriever_raw", text=raw_chunk_ids[:2000])

    # Parse chosen ids (robust against ```
    chosen_ids: List[str] = []
    parsed = _strip_code_fences(raw_chunk_ids)

    try:
        chosen_ids_any = json.loads(parsed)
        if isinstance(chosen_ids_any, list):
            chosen_ids = [str(x) for x in chosen_ids_any]
    except Exception:
        chosen_ids = []

    chosen = [c for c in chunks if c[0] in set(chosen_ids)]
    if not chosen:
        chosen = chunks[:5]
    tracer.log("retriever_parsed", chosen_ids=chosen_ids[:20], parsed_ok=bool(chosen_ids))

    context_block = "\n\n".join([f"[{cid}]\n{txt}" for cid, txt in chosen])

    # Deterministic context logging for strict judging
    tracer.log(
        "rag_context",
        context_block_sha256=_sha256(context_block),
        context_block_preview=context_block[:2000],
        chosen=[{"id": cid, "text": txt} for cid, txt in chosen],
    )
    tracer.log("chosen_chunks", ids=[cid for cid, _ in chosen])

    answer_prompt = (
        "Answer the question using ONLY the provided context.\n"
        "Cite chunk ids like [doc.txt#chunkN] after the sentence that uses it.\n"
        "If not answerable from context, say: Not enough information in provided documents.\n\n"
        f"Question:\n{user_prompt}\n\n"
        f"Context:\n{context_block}\n"
    )

    draft = _call(solver, answer_prompt, temperature=0.2)
    tracer.log("solver_draft", model=solver["model"], text=draft[:4000])

    final = draft
    if verifier:
        verify_prompt = (
            "You are a strict verifier.\n"
            "Rules:\n"
            "1) Delete any sentence/bullet that is not directly supported by the provided Context.\n"
            "2) Every non-trivial sentence MUST include at least one chunk-id citation like [doc.txt#chunkN].\n"
            "3) Do NOT add new sections or general knowledge.\n"
            "4) If a part of the question cannot be answered from Context, output exactly: "
            "'Not enough information in provided documents.' for that part.\n"
            "Return ONLY the final answer.\n\n"
            f"Question:\n{user_prompt}\n\n"
            f"Answer:\n{draft}\n\n"
            f"Context:\n{context_block}\n"
            )
        final = _call(verifier, verify_prompt, temperature=0.1)
        tracer.log("verifier_final", model=verifier["model"], text=final[:4000])

    tracer.set_result(success=True, final_answer=final)
    trace_path = tracer.flush()
    return {"task_id": task_id, "answer": final, "trace": str(trace_path)}
