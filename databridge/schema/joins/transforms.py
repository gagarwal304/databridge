from __future__ import annotations

import re
from typing import Any, Callable

Transform = Callable[[Any], str]

# Each entry: (name, forward_fn)
# Applied to values from column A before comparing against column B values.
TRANSFORM_GRAMMAR: list[tuple[str, Transform]] = [
    ("identity", lambda x: str(x).strip()),
    ("lowercase", lambda x: str(x).strip().lower()),
    ("extract_digits", lambda x: re.sub(r"\D", "", str(x))),
    ("strip_prefix_3", lambda x: str(x)[3:] if len(str(x)) > 3 else str(x)),
    ("strip_prefix_4", lambda x: str(x)[4:] if len(str(x)) > 4 else str(x)),
    ("strip_prefix_5", lambda x: str(x)[5:] if len(str(x)) > 5 else str(x)),
    ("zfill_5", lambda x: str(int(re.sub(r"\D", "", str(x)) or 0)).zfill(5)),
    ("zfill_7", lambda x: str(int(re.sub(r"\D", "", str(x)) or 0)).zfill(7)),
    ("remove_separators", lambda x: re.sub(r"[-_\s]", "", str(x))),
]


def detect_transform(
    values_a: list[Any],
    values_b: list[Any],
    overlap_threshold: float = 0.70,
) -> tuple[str | None, float]:
    """
    Try each transform on values_a and check overlap against values_b (as strings).
    Returns (transform_name, overlap_ratio) for the best match, or (None, 0.0).
    """
    set_b = {str(v).strip() for v in values_b if v is not None}
    if not set_b:
        return None, 0.0

    best_name: str | None = None
    best_ratio = 0.0

    for name, fn in TRANSFORM_GRAMMAR:
        transformed = set()
        for v in values_a:
            if v is None:
                continue
            try:
                transformed.add(fn(v))
            except Exception:
                continue
        if not transformed:
            continue
        overlap = len(transformed & set_b) / len(transformed)
        if overlap > best_ratio:
            best_ratio = overlap
            best_name = name

    if best_ratio >= overlap_threshold:
        return best_name, best_ratio
    return None, best_ratio
