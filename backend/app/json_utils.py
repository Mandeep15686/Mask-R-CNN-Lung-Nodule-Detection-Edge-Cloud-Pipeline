"""Helpers for emitting strict, standards-compliant JSON responses."""
import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Replace non-finite numeric values with null recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    # Support NumPy scalar values without requiring NumPy in this module.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except (TypeError, ValueError):
            pass
    return value
