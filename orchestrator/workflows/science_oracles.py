from typing import Any, Dict, Optional, Callable, List
import math
import re


# ============================
# Shared extraction helpers
# ============================

def _extract_height(prompt: str) -> float:
    """Extract height in meters from prompt."""
    p = (prompt or "").lower()
    patterns = [
        r"(?:height|h)\s*(?:of\s*)?(?:=)?\s*([0-9]+(?:\.[0-9]+)?)\s*m\b",
        r"from\s+([0-9]+(?:\.[0-9]+)?)\s*m\b",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return float(m.group(1))
    raise ValueError("Could not extract height (m)")


def _extract_mu_k(prompt: str) -> float:
    """Extract kinetic friction coefficient μk."""
    p = (prompt or "").lower()
    patterns = [
        r"(?:μk|mu_k|mu k|kinetic friction coefficient)\s*(?:=)?\s*([0-9]+(?:\.[0-9]+)?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return float(m.group(1))
    raise ValueError("Could not extract mu_k")


def _extract_mass(prompt: str) -> float:
    """Extract mass in kg."""
    p = (prompt or "").lower()
    patterns = [
        r"(?:mass|m)\s*(?:of\s*)?(?:=)?\s*([0-9]+(?:\.[0-9]+)?)\s*kg\b",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return float(m.group(1))
    raise ValueError("Could not extract mass (kg)")


def _extract_velocity(prompt: str, var_name: str = "v") -> float:
    """Extract velocity in m/s; var_name can be 'v', 'v0', 'u', etc."""
    p = (prompt or "").lower()
    patterns = [
        rf"(?:{var_name}|velocity)\s*(?:of\s*)?(?:=)?\s*([0-9]+(?:\.[0-9]+)?)\s*m/s\b",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return float(m.group(1))
    raise ValueError(f"Could not extract velocity {var_name} (m/s)")


def _extract_acceleration(prompt: str) -> float:
    """Extract acceleration in m/s^2."""
    p = (prompt or "").lower()
    patterns = [
        r"(?:acceleration|a)\s*(?:of\s*)?(?:=)?\s*([0-9]+(?:\.[0-9]+)?)\s*m/s",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return float(m.group(1))
    raise ValueError("Could not extract acceleration (m/s^2)")


def _extract_distance(prompt: str) -> float:
    """Extract distance in meters."""
    p = (prompt or "").lower()
    patterns = [
        r"(?:distance|d)\s*(?:of\s*)?(?:=)?\s*([0-9]+(?:\.[0-9]+)?)\s*m\b",
    ]
    for pat in patterns:
        m = re.search(pat, p)
        if m:
            return float(m.group(1))
    raise ValueError("Could not extract distance (m)")


# ============================
# Oracle handlers
# ============================

def _oracle_ramp_friction(prompt: str, g: float = 9.8) -> Dict[str, Any]:
    """
    Template: frictionless ramp of height h, then horizontal surface with μk.
    Asks: speed at bottom, distance to stop.
    """
    p_lower = (prompt or "").lower()
    
    # Strict gating: require both ramp+friction keywords
    if not re.search(r"\b(ramp|incline|slope)\b", p_lower):
        raise ValueError("Not a ramp problem")
    if not re.search(r"\b(friction|μk|mu_k)\b", p_lower):
        raise ValueError("No friction mentioned")
    if not re.search(r"\b(speed|velocity)\b.*\b(bottom|base)\b", p_lower):
        raise ValueError("Does not ask for speed at bottom")

    h = _extract_height(prompt)
    mu = _extract_mu_k(prompt)

    v = math.sqrt(2 * g * h)
    d = (v * v) / (2 * mu * g)

    return {
        "name": "ramp_friction_v_and_stop_distance",
        "final": {
            "v_bottom": f"{v:.2f} m/s",
            "distance": f"{d:.2f} m",
        },
        "final_text": f"v_bottom: {v:.2f} m/s\ndistance: {d:.2f} m",
        "parsed": {"h": h, "mu_k": mu, "g": g, "v_bottom": v, "distance": d},
    }


def _oracle_free_fall_speed(prompt: str, g: float = 9.8) -> Dict[str, Any]:
    """
    Template: object dropped/falls from height h (no friction), asks for speed at impact.
    """
    p_lower = (prompt or "").lower()
    
    # Gating: must mention fall/drop and ask for speed
    if not re.search(r"\b(fall|falls|dropped|drop|free[- ]?fall)\b", p_lower):
        raise ValueError("Not a free-fall prompt")
    if not re.search(r"\b(speed|velocity)\b", p_lower):
        raise ValueError("Free-fall oracle requires speed/velocity request")

    h = _extract_height(prompt)
    v = math.sqrt(2 * g * h)

    return {
        "name": "free_fall_speed_from_height",
        "final": {
            "v": f"{v:.2f} m/s",
        },
        "final_text": f"v: {v:.2f} m/s",
        "parsed": {"h": h, "g": g, "v": v},
    }


def _oracle_constant_accel_kinematics(prompt: str) -> Dict[str, Any]:
    """
    Template: constant acceleration from rest (or initial velocity), asks for final velocity or distance.
    Uses: v^2 = v0^2 + 2*a*d or d = (v^2 - v0^2)/(2*a)
    """
    p_lower = (prompt or "").lower()

    # Gating: must mention acceleration/deceleration and distance or velocity
    if not re.search(r"\b(accelerat|decelerat|constant.*a)\b", p_lower):
        raise ValueError("Not a constant-acceleration problem")

    # Try to extract known variables
    v0 = 0.0  # default: starts from rest
    try:
        v0 = _extract_velocity(prompt, var_name="v0")
    except:
        try:
            v0 = _extract_velocity(prompt, var_name="u")
        except:
            pass  # assume rest

    a = _extract_acceleration(prompt)

    # Determine what's asked: final velocity or distance?
    if re.search(r"\b(final.*speed|final.*velocity|speed.*after|velocity.*after)\b", p_lower):
        # Need distance to compute v
        d = _extract_distance(prompt)
        v = math.sqrt(v0 * v0 + 2 * a * d)
        return {
            "name": "constant_accel_final_velocity",
            "final": {"v": f"{v:.2f} m/s"},
            "final_text": f"v: {v:.2f} m/s",
            "parsed": {"v0": v0, "a": a, "d": d, "v": v},
        }
    elif re.search(r"\b(distance|how far)\b", p_lower):
        # Need final velocity to compute distance
        v = _extract_velocity(prompt, var_name="v")
        d = (v * v - v0 * v0) / (2 * a)
        return {
            "name": "constant_accel_distance",
            "final": {"distance": f"{d:.2f} m"},
            "final_text": f"distance: {d:.2f} m",
            "parsed": {"v0": v0, "a": a, "v": v, "d": d},
        }
    else:
        raise ValueError("Constant-accel oracle: unclear what's being asked")


# ============================
# Registry
# ============================

ORACLES: List[Callable[[str], Dict[str, Any]]] = [
    _oracle_ramp_friction,
    _oracle_free_fall_speed,
    _oracle_constant_accel_kinematics,
]


def try_science_oracles(user_prompt: str) -> Optional[Dict[str, Any]]:
    """
    Try all registered oracles in order; return the first match or None.
    """
    for fn in ORACLES:
        try:
            return fn(user_prompt)
        except Exception:
            continue
    return None
