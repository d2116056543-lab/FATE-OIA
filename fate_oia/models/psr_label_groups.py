from __future__ import annotations

DEFAULT_ACTION_LABELS = [0, 1, 2, 3]
DEFAULT_TAIL_REASONS = [5, 6, 9, 11, 12, 14]
DEFAULT_WEAK_TAIL_REASONS = [8, 10, 13]
DEFAULT_ALL_REASONS = list(range(21))


def normalize_label_groups(config: dict | None = None) -> dict[str, list[int]]:
    cfg = config or {}
    return {
        "action": list(cfg.get("action", DEFAULT_ACTION_LABELS)),
        "tail_reason": list(cfg.get("tail_reason", DEFAULT_TAIL_REASONS)),
        "weak_tail_reason": list(cfg.get("weak_tail_reason", DEFAULT_WEAK_TAIL_REASONS)),
        "all_reason": list(cfg.get("all_reason", DEFAULT_ALL_REASONS)),
    }


def reason_is_tail(index: int, groups: dict[str, list[int]] | None = None) -> bool:
    g = normalize_label_groups(groups)
    return int(index) in set(g["tail_reason"]) or int(index) in set(g["weak_tail_reason"])
