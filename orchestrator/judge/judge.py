from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
import json

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")  # captures inside [ ... ]
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

def _get_last_event(events: List[Dict[str, Any]], kind: str) -> Optional[Dict[str, Any]]:
    for e in reversed(events):
        if e.get("kind") == kind:
            return e
    return None


def _get_event(events: List[Dict[str, Any]], kind: str) -> Optional[Dict[str, Any]]:
    for e in events:
        if e.get("kind") == kind:
            return e
    return None


def _extract_citations(text: str) -> Set[str]:
    return {m.group(1).strip() for m in _CITATION_RE.finditer(text or "") if m.group(1).strip()}


def _is_nonclaim_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True

    low = s.lower()

    # Meta/commentary lines that should never require citations
    if low.startswith(
        (
            "note:",
            "notes:",
            "rule ",
            "rules:",
            "format:",
            "citation:",
            "citations:",
            "as per ",
            "the answer is",
            "here is the answer",
            "answer:",
            "answer to",
        )
    ):
        return True
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]") and len(s) < 80):
        return True

    # separators / headings
    if s.startswith("#") or s in {"---", "-", "*"}:
        return True
    if len(s) < 120 and s.endswith(":"):
        return True

    # any question line
    if s.endswith("?"):
        return True

    # numbered prompt lines like: 1. ...
    if re.match(r"^\d+\.\s", s):
        return True

    # quoted prompt lines
    if (s.startswith("“") and s.endswith("”")) or (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return True

    return False


def _merge_citation_only_lines(text: str) -> str:
    """
    If the model outputs:
      Claim sentence.
      [doc#chunk0]
    merge into:
      Claim sentence. [doc#chunk0]
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        cites = _extract_citations(s)
        if cites and (s.startswith("[") and s.endswith("]")) and out:
            out[-1] = out[-1].rstrip() + " " + s
            continue

        out.append(s)
    return "\n".join(out)

def _parse_verifier_status(events: List[Dict[str, Any]]) -> Optional[str]:
    # Prefer last verifier-like event
    verifier_evt = _get_last_event(events, "verifier") or _get_last_event(events, "verifier_recheck")
    payload = (verifier_evt or {}).get("payload", {}) or {}
    # payload might be dict OR might be {"text": "...json..."} depending on logger
    if isinstance(payload, dict):
        # Case A: you logged parsed JSON directly into payload
        if isinstance(payload.get("status"), str):
            return payload.get("status").strip().lower()
        # Case B: you logged text that contains JSON
        txt = payload.get("text")
        if isinstance(txt, str) and txt.strip().startswith("{"):
            try:
                obj = json.loads(txt)
                if isinstance(obj.get("status"), str):
                    return obj["status"].strip().lower()
            except Exception:
                return None
    return None

def _looks_like_units(text: str) -> bool:
    return bool(re.search(r"(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(m/s|m/s\^2|kg|j|n|pa|hz|cm|mm|m|s|h)\b", text.lower()))

def _count_numbers(text: str) -> int:
    return len(re.findall(r"\d+(?:\.\d+)?", text or ""))



def _looks_like_numeric_answer(text: str) -> bool:
    # crude: has at least one digit and not just a year; good enough for v1 scoring
    if not text:
        return False
    return bool(re.search(r"\d", text))


def judge_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    workflow = trace.get("workflow")
    events: List[Dict[str, Any]] = trace.get("events", []) or []
    final_answer = trace.get("final_answer") or trace.get("answer") or ""

    # --- CODING: objective via pytest ---
    if workflow == "coding":
        pytest_event = _get_last_event(events, "pytest")
        payload = (pytest_event or {}).get("payload", {}) or {}

        exit_code = payload.get("exitcode", payload.get("exit_code", None))
        passed = (exit_code == 0)

        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "reason": f"pytest exit_code={exit_code}",
            "signals": {
                "exit_code": exit_code,
                "has_pytest_event": pytest_event is not None,
            },
        }

    # --- RAG_QA: strict grounding judge using rag_context ---
    if workflow == "rag_qa":
        final_answer_norm = _merge_citation_only_lines(final_answer)

        cited_all = _extract_citations(final_answer_norm)
        abstained_global = "not enough information" in final_answer_norm.lower()

        chosen_evt = _get_event(events, "chosen_chunks")
        chosen_ids_logged = set(((chosen_evt or {}).get("payload", {}) or {}).get("ids", []) or [])

        ctx_evt = _get_event(events, "rag_context")
        ctx_payload = (ctx_evt or {}).get("payload", {}) or {}
        chosen_list = ctx_payload.get("chosen", []) or []
        chosen_ids_ctx = {c.get("id") for c in chosen_list if isinstance(c, dict) and c.get("id")}

        context_preview = (ctx_payload.get("context_block_preview") or "").strip()

        invalid_citations = set()
        if cited_all:
            if chosen_ids_logged:
                invalid_citations |= {c for c in cited_all if c not in chosen_ids_logged}
            if chosen_ids_ctx:
                invalid_citations |= {c for c in cited_all if c not in chosen_ids_ctx}
            if context_preview:
                invalid_citations |= {c for c in cited_all if c not in context_preview}

        # Must cite unless abstaining
        if not abstained_global and not cited_all:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "no citations and did not abstain",
                "signals": {
                    "abstained": abstained_global,
                    "num_citations": 0,
                    "has_rag_context_event": ctx_evt is not None,
                },
            }

        lines = [ln.strip() for ln in _SENT_SPLIT.split(final_answer_norm) if ln.strip()]
        uncited_claim_lines: List[str] = []
        invalid_cite_lines: List[Dict[str, Any]] = []

        for ln in lines:
            if _is_nonclaim_line(ln):
                continue
            if "not enough information" in ln.lower():
                continue

            cites = _extract_citations(ln)
            if not cites:
                uncited_claim_lines.append(ln[:200])
                continue

            bad = set()
            if chosen_ids_logged:
                bad |= {c for c in cites if c not in chosen_ids_logged}
            if chosen_ids_ctx:
                bad |= {c for c in cites if c not in chosen_ids_ctx}
            if context_preview:
                bad |= {c for c in cites if c not in context_preview}

            if bad:
                invalid_cite_lines.append({"line": ln[:200], "bad": sorted(bad)})

        passed = bool(
            abstained_global
            or (not invalid_citations and not uncited_claim_lines and not invalid_cite_lines)
        )
        score = 1.0 if passed else 0.0
        reason = "rag grounded" if passed else "rag has uncited/invalid claim lines"

        return {
            "passed": passed,
            "score": score,
            "reason": reason,
            "signals": {
                "abstained": abstained_global,
                "cited": sorted(cited_all),
                "invalid_citations": sorted(invalid_citations),
                "uncited_claim_lines_n": len(uncited_claim_lines),
                "invalid_cite_lines_n": len(invalid_cite_lines),
                "uncited_claim_lines_preview": uncited_claim_lines[:5],
                "invalid_cite_lines_preview": invalid_cite_lines[:5],
                "has_chosen_chunks_event": chosen_evt is not None,
                "has_rag_context_event": ctx_evt is not None,
                "chosen_ids_logged_n": len(chosen_ids_logged),
                "chosen_ids_ctx_n": len(chosen_ids_ctx),
                "has_context_preview": bool(context_preview),
            },
        }

    # --- REASONING / SCIENCE ---

    if workflow in ("reasoning", "science"):
        has_solver_final = _get_event(events, "solver_final") is not None
        has_solver_draft = _get_event(events, "solver_draft") is not None

        has_verifier = (_get_event(events, "verifier") is not None) or (_get_event(events, "verifier_recheck") is not None)
        has_critic = _get_event(events, "critic") is not None

        verifier_status = _parse_verifier_status(events)

        if verifier_status == "reject":
            return {
                "passed": False,
                "score": 0.0,
                "reason": "verifier_reject",
                "signals": {
                    "has_solver_draft": has_solver_draft,
                    "has_solver_final": has_solver_final,
                    "has_verifier": has_verifier,
                    "verifier_status": verifier_status,
                    "verifier_parse_failed": bool(has_verifier and verifier_status is None),
                    "answer_len": len(final_answer.strip()),
                },
            }

        verifier_accept = (verifier_status == "accept")

        has_numeric = _looks_like_numeric_answer(final_answer)
        has_units = _looks_like_units(final_answer)

        passed = bool(
            verifier_accept
            or ((has_solver_final or has_solver_draft) and has_numeric and has_units)
        )

        prompt = trace.get("prompt", "") or ""
        multipart = ("1)" in prompt and "2)" in prompt) or (prompt.count("?") >= 2)

        score = 0.0
        reason = "failed_contract"

        if (workflow == "science") and (not verifier_accept) and multipart:
            if _count_numbers(final_answer) < 2:
                passed = False
                reason = "incomplete_numeric_answer"

        if passed:
            score = 0.6
            reason = "contract_ok"

            if verifier_accept:
                score = max(score, 0.95)
                reason = "verifier_accept"

            if has_verifier:
                score = min(1.0, score + 0.03)
            if has_critic:
                score = min(1.0, score + 0.02)

        return {
            "passed": bool(passed),
            "score": float(score),
            "reason": reason,
            "signals": {
                "verifier_parse_failed": bool(has_verifier and verifier_status is None),
                "multipart": multipart,
                "has_solver_draft": has_solver_draft,
                "has_solver_final": has_solver_final,
                "has_verifier": has_verifier,
                "has_critic": has_critic,
                "verifier_status": verifier_status,
                "verifier_accept": verifier_accept,
                "has_numeric": has_numeric,
                "has_units": has_units,
                "answer_len": len(final_answer.strip()),
            },
        }
    # --- FALLBACK ---
    passed = bool(trace.get("success", False))
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "reason": "trace.success (fallback)",
        "signals": {"workflow": workflow},
    }

