from __future__ import annotations

import html
import json
from pathlib import Path

from fate_oia.acpr_interactflow.artifacts import write_json
from fate_oia.explain.acpr_interactflow_faithfulness import summarize_intervention_audit


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_atlas(
    case_dirs: list[str | Path],
    output_html: str | Path,
    metrics_path: str | Path | None = None,
    intervention_path: str | Path | None = None,
) -> dict:
    cases = []
    for item in case_dirs:
        case_dir = Path(item)
        case_json = case_dir / "decision_ledger.json"
        if not case_json.exists():
            case_json = case_dir / "case_source.json"
        case = _read_json(case_json)
        if case:
            cases.append({"dir": str(case_dir), "case": case})

    metrics = _read_json(Path(metrics_path)) if metrics_path else {}
    intervention = _read_json(Path(intervention_path)) if intervention_path else {}
    faithfulness = summarize_intervention_audit(intervention) if intervention else {"available": False}

    action_rows = []
    for item in cases:
        case = item["case"]
        action_rows.append(
            "<tr>"
            f"<td>{html.escape(str(case.get('file_name', case.get('sample_id', 'unknown'))))}</td>"
            f"<td>{case.get('gt_action')}</td>"
            f"<td>{case.get('pred_action')}</td>"
            f"<td>{float(case.get('identity_check_max_abs', 0.0)):.6g}</td>"
            f"<td><a href='{html.escape(Path(item['dir']).name)}/report.html'>case report</a></td>"
            "</tr>"
        )
    action_table = "\n".join(action_rows)
    metrics_html = html.escape(json.dumps(metrics, indent=2, ensure_ascii=False)[:6000])
    faith_html = html.escape(json.dumps(faithfulness, indent=2, ensure_ascii=False))
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""
<html><body>
<h1>ACPR-InteractFlow++ Dataset Atlas</h1>
<h2>Run Metrics</h2><pre>{metrics_html}</pre>
<h2>Model-Level Counterfactual Dependence</h2><pre>{faith_html}</pre>
<h2>Decision Ledger Cases</h2>
<table border="1"><tr><th>case</th><th>gt</th><th>pred</th><th>identity error</th><th>link</th></tr>
{action_table}
</table>
<p>No manual boxes or fabricated effects are used; all rows are sourced from saved eval/intervention artifacts.</p>
</body></html>
""",
        encoding="utf-8",
    )
    manifest = {
        "case_count": len(cases),
        "metrics_path": str(metrics_path) if metrics_path else None,
        "intervention_path": str(intervention_path) if intervention_path else None,
        "faithfulness": faithfulness,
        "output_html": str(output),
        "manual_boxes_forbidden": True,
        "fabricated_effects_forbidden": True,
    }
    write_json(output.with_suffix(".json"), manifest)
    return manifest
