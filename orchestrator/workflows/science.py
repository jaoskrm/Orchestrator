from typing import Any, Dict
import json
import math
import re

from orchestrator.llm_clients import OllamaClient
from orchestrator.storage.traces import TraceWriter


def _call(worker: Dict[str, Any], prompt: str, temperature: float = 0.2) -> str:
    if worker["provider"] == "ollama":
        return OllamaClient().generate(model=worker["model"], prompt=prompt, temperature=temperature)
    raise ValueError(f"Unsupported provider: {worker['provider']}")


def _extract_first_json_object(raw: str) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from a raw LLM response.
    Accepts cases where the model adds extra text before/after JSON.
    """
    s = (raw or "").strip()
    if not s:
        raise ValueError("Empty verifier output")

    # Fast path
    if s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            pass

    in_str = False
    esc = False
    depth = 0
    start = -1

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                if depth == 0 and start != -1:
                    return json.loads(s[start : i + 1])

    raise ValueError(f"Verifier did not return valid JSON. Raw: {s[:400]}")


def _format_final(final_obj: Dict[str, Any]) -> str:
    """
    Convert verifier 'final' object into a strict, short final answer.
    """
    v = (final_obj or {}).get("v_bottom", "")
    d = (final_obj or {}).get("distance", "")
    return f"v_bottom: {v}\ndistance: {d}".strip()


def run_science_workflow(task_id: str, user_prompt: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    tracer = TraceWriter(task_id, "science", user_prompt, decision)

    workers = decision.get("workers", []) or []
    solver = next((w for w in workers if w.get("role") == "solver"), None)
    verifier = next((w for w in workers if w.get("role") in ("verifier", "critic")), None)

    if not solver:
        raise ValueError("science workflow requires a solver worker")
    if not verifier:
        raise ValueError("science workflow requires a verifier/critic worker (enforced by router)")

    controls = decision.get("controls", {}) or {}
    # Science semantics:
    # - allow one automatic "repair" if verifier rejects (always)
    # - if use_debate=True, allow extra repair loops up to max_rounds (capped)
    use_debate = bool(controls.get("use_debate", False))
    max_rounds = int(controls.get("max_rounds", 2))
    max_rounds = max(1, min(max_rounds, 4))

    solve_prompt = (
        "You are a science problem solver (math/physics/chem).\n"
        "Solve the problem carefully.\n"
        "Do NOT be verbose.\n"
        "At the end, include a short line 'Final answer:' with the key numeric results + units.\n\n"
        f"Problem:\n{user_prompt}\n"
    )

    verifier_prompt_template = (
    "You are a NUMERIC VERIFIER for math/physics/chem solutions.\n\n"
    "You MUST recompute every requested numerical result independently from the Problem.\n"
    "Do NOT trust the candidate’s arithmetic or intermediate steps.\n\n"
    "Return ONLY valid JSON (no markdown, no commentary) with this schema:\n"
    "{{\n"
    '  "status": "accept" | "reject",\n'
    '  "errors": ["..."],\n'
    '  "final": {{\n'
    '    "v_bottom": "<number> m/s",\n'
    '    "distance": "<number> m"\n'
    "  }}\n"
    "}}\n\n"
    "Rules:\n"
    "- Use g = 9.8 m/s^2 unless the problem specifies otherwise.\n"
    "- Recompute v_bottom and distance from scratch.\n"
    "- Compute relative error for each final value: |candidate - yours| / max(|yours|, 1e-9).\n"
    "- If any relative error > 0.02 (2%), set status=\"reject\" and list the mismatch.\n"
    "- If you did not recompute (or are unsure), set status=\"reject\".\n"
    "- Always fill the 'final' fields with YOUR recomputed best values + units.\n"
    "- Keep 'errors' short (max 3 bullets). Do NOT write a full derivation.\n\n"
    "Problem:\n{problem}\n\n"
    "Candidate solution:\n{candidate}\n"
)



    # Round 1: draft
    draft = _call(solver, solve_prompt, temperature=0.2)
    tracer.log("solver_draft", model=solver["model"], text=draft[:2000])

    # Round 1: verify
    vraw = _call(
        verifier,
        verifier_prompt_template.format(problem=user_prompt, candidate=draft),
        temperature=0.1,
    )
    tracer.log("verifier", model=verifier["model"], text=vraw[:2000])

    vobj = _extract_first_json_object(vraw)
    status = str(vobj.get("status", "")).strip().lower()
    final_obj = vobj.get("final") if isinstance(vobj.get("final"), dict) else {}

    # If accepted, terminate immediately with short final
    if status == "accept":
        final_text = _format_final(final_obj)
        tracer.log("solver_final", model=solver["model"], text=final_text[:2000])
        tracer.set_result(success=True, final_answer=final_text)
        trace_path = tracer.flush()
        return {"task_id": task_id, "answer": final_text, "trace": str(trace_path)}

    # Otherwise: at least one repair attempt (even if use_debate=False)
    # Remaining rounds: if max_rounds=2 => exactly one repair loop.
    rounds_left = max_rounds - 1  # already used 1 solver call
    repair_attempts_allowed = 1 if not use_debate else max(1, rounds_left)

    current_candidate = draft
    last_final_text = _format_final(final_obj)

    for attempt in range(1, repair_attempts_allowed + 1):
        errors = vobj.get("errors", [])
        if not isinstance(errors, list):
            errors = [str(errors)] if errors else []

        revise_prompt = (
            "Revise your answer based on the verifier feedback.\n"
            "Output ONLY the final answers in exactly this format:\n"
            "v_bottom: <number> m/s\n"
            "distance: <number> m\n\n"
            f"Problem:\n{user_prompt}\n\n"
            f"Verifier errors:\n- " + "\n- ".join(str(e) for e in errors) + "\n\n"
            f"Verifier suggested final (use if correct):\n{last_final_text}\n"
        )

        revised = _call(solver, revise_prompt, temperature=0.2)
        tracer.log("solver_revise", model=solver["model"], text=revised[:2000])

        vraw2 = _call(
            verifier,
            verifier_prompt_template.format(problem=user_prompt, candidate=revised),
            temperature=0.1,
        )
        tracer.log("verifier_recheck", model=verifier["model"], text=vraw2[:2000])

        vobj = _extract_first_json_object(vraw2)
        status2 = str(vobj.get("status", "")).strip().lower()
        final_obj2 = vobj.get("final") if isinstance(vobj.get("final"), dict) else {}
        last_final_text = _format_final(final_obj2)

        if status2 == "accept":
            break

        # If not debating, do not loop further
        if not use_debate:
            break

    # Always use verifier's latest "final" as the canonical final answer (short + structured)
    tracer.log("solver_final", model=solver["model"], text=last_final_text[:2000])
    tracer.set_result(success=True, final_answer=last_final_text)
    trace_path = tracer.flush()
    return {"task_id": task_id, "answer": last_final_text, "trace": str(trace_path)}
