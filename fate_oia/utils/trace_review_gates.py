FORBIDDEN_SUPERVISOR_TERMS = ["Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup", "hidden cmd"]
FORBIDDEN_PROXY_TERMS = ["target_deleted_reason = reason_logits -", "context_only_reason = base[\"reason_logits\"]", "evidence_only_reason = reason_gate * reason_delta", "cf_is_proxy=True"]


def scan_forbidden(paths, terms):
    from pathlib import Path
    found = {}
    for raw in paths:
        p = Path(raw)
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore")
            hits = [t for t in terms if t in text]
            if hits:
                found[str(p)] = hits
    return found
