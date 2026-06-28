from __future__ import annotations

from pathlib import Path

FORBIDDEN_PATTERNS = [
    "frozen_run_c", "FrozenRunC", "run_c_logits", "cached_logits",
    "distill_from_checkpoint", "tail_residual_adapter", "ComplementaryLogitFusionAdapter",
    "PMI", "pmi", "cooccurrence", "co-occurrence", "label_graph", "graph_delta_to_logits",
    "MoE", "moe", "expert", "Expert", "router", "Router", "selector", "Selector",
    "evidence_memory", "feature_cache_enabled: true", "token_compression: keep_merge",
    "checkpoint_best_val", "best_selection_split: val", "eval_splits: val",
    "Start-Process", "Start-Job", "nohup", "hidden", "scheduled task", "daemon",
]


def scan_paths(paths: list[str | Path], *, allow_audit_files: bool = True) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for path in paths:
        p = Path(path)
        if not p.exists() or p.is_dir():
            continue
        if allow_audit_files and ("audit" in p.name or "forbidden_scan" in p.name):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        hits = [pat for pat in FORBIDDEN_PATTERNS if pat in text]
        if hits:
            results[str(p)] = hits
    return results
