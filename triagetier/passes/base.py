from __future__ import annotations

_TRUTHY = {"yes", "true", "present", "positive", "abnormal"}


def parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields


def is_positive(value: str) -> bool:
    return value.strip().lower() in _TRUTHY
