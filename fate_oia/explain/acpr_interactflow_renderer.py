from __future__ import annotations

import html
from pathlib import Path

from PIL import Image, ImageDraw

from fate_oia.acpr_interactflow.artifacts import write_json

ACTION_NAMES = ["maintain_speed", "reduce_speed", "stop_car"]


def _draw_waterfall(path: Path, case: dict) -> None:
    final = case.get("final_logits", [])
    global_logits = case.get("global_logits", [])
    flow = case.get("flow_delta_logits", [])
    calibration = case.get("calibration_delta", [])
    width, height = 760, 280
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), "Dynamic Interaction Decision Ledger", fill="black")
    zero_x = 360
    scale = 70.0
    for i, name in enumerate(ACTION_NAMES[: len(final)]):
        y = 52 + i * 64
        draw.text((16, y), name, fill="black")
        draw.line((zero_x, y - 4, zero_x, y + 48), fill=(180, 180, 180))
        parts = [
            ("global", global_logits[i] if i < len(global_logits) else 0.0, (70, 120, 220)),
            ("flow", flow[i] if i < len(flow) else 0.0, (230, 120, 40)),
            ("calib", calibration[i] if i < len(calibration) else 0.0, (150, 90, 180)),
            ("final", final[i], (40, 150, 80)),
        ]
        for j, (label, value, color) in enumerate(parts):
            yy = y + j * 12
            x2 = int(zero_x + float(value) * scale)
            x1, x2 = min(zero_x, x2), max(zero_x, x2)
            draw.rectangle((x1, yy, x2, yy + 10), fill=color)
            draw.text((520, yy - 2), f"{label}: {float(value):+.3f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def render_case(case: dict, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "case_source.json", case)
    _draw_waterfall(out / "decision_waterfall.png", case)
    escaped_name = html.escape(str(case.get("file_name", case.get("sample_id", "unknown"))))
    identity = float(case.get("identity_check_max_abs", 0.0))
    report = f"""
<html><body>
<h1>ACPR-InteractFlow++ Dynamic Interaction Decision Ledger</h1>
<p>case={escaped_name}</p>
<p>gt_action={case.get('gt_action')} pred_action={case.get('pred_action')}</p>
<p>ledger_identity_max_abs={identity:.8f}</p>
<img src="decision_waterfall.png" width="760">
<h2>Tensor lineage</h2>
<pre>{html.escape(str(case.get('tensor_lineage', 'global + flow_delta + calibration = final')))}</pre>
</body></html>
"""
    (out / "report.html").write_text(report, encoding="utf-8")
