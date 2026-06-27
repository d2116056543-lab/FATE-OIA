from __future__ import annotations

import argparse
import html
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from fate_oia.acpr_interactflow.artifacts import write_json
from fate_oia.explain.acpr_interactflow_renderer import render_case

ACTION_NAMES = ["maintain_speed", "reduce_speed", "stop_car"]


def _safe_load_tensor(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _draw_contribution_png(path: Path, final_logits: torch.Tensor, global_logits: torch.Tensor, flow_delta: torch.Tensor, calibration_delta: torch.Tensor) -> None:
    width, height = 720, 260
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), "Dynamic Interaction Decision Ledger", fill="black")
    zero_x = 360
    bar_h = 18
    scale = 70.0
    for i, name in enumerate(ACTION_NAMES[: final_logits.numel()]):
        y = 52 + i * 58
        draw.text((16, y), name, fill="black")
        draw.line((zero_x, y - 6, zero_x, y + 36), fill=(180, 180, 180))
        parts = [
            ("global", float(global_logits[i]), (70, 120, 220)),
            ("flow", float(flow_delta[i]), (230, 120, 40)),
            ("calib", float(calibration_delta[i]), (150, 90, 180)),
            ("final", float(final_logits[i]), (40, 150, 80)),
        ]
        for j, (label, value, color) in enumerate(parts):
            yy = y + j * (bar_h + 1)
            x2 = int(zero_x + value * scale)
            x1, x2 = min(zero_x, x2), max(zero_x, x2)
            draw.rectangle((x1, yy, x2, yy + bar_h), fill=color)
            draw.text((500, yy), f"{label}: {value:+.3f}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_cases", type=int, default=8)
    args = parser.parse_args()
    metrics_path = Path(args.metrics)
    run_dir = metrics_path.parent
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    final_logits = _safe_load_tensor(run_dir / "logits_action_test.pt")
    global_logits = _safe_load_tensor(run_dir / "logits_action_global_test.pt")
    flow_delta = _safe_load_tensor(run_dir / "logits_action_flow_delta_test.pt")
    calibration_delta = _safe_load_tensor(run_dir / "logits_action_calibration_delta_test.pt")
    state_contrib = _safe_load_tensor(run_dir / "ledger_gated_state_contributions_test.pt")
    labels = _safe_load_tensor(run_dir / "labels_action_test.pt")
    file_names_path = run_dir / "file_names_test.json"
    file_names = []
    if file_names_path.exists():
        import json

        file_names = json.loads(file_names_path.read_text(encoding="utf-8"))

    required = {
        "logits_action_test.pt": final_logits is not None,
        "logits_action_global_test.pt": global_logits is not None,
        "logits_action_flow_delta_test.pt": flow_delta is not None,
        "logits_action_calibration_delta_test.pt": calibration_delta is not None,
        "ledger_gated_state_contributions_test.pt": state_contrib is not None,
        "labels_action_test.pt": labels is not None,
        "file_names_test.json": bool(file_names),
    }
    available = all(required.values())
    cases = []
    if available:
        count = min(args.max_cases, final_logits.shape[0])
        for i in range(count):
            case_dir = out / f"case_{i:04d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            case = {
                "index": i,
                "file_name": file_names[i] if i < len(file_names) else str(i),
                "gt_action": int(labels[i]),
                "pred_action": int(final_logits[i].argmax()),
                "action_names": ACTION_NAMES,
                "final_logits": final_logits[i].tolist(),
                "global_logits": global_logits[i].tolist(),
                "flow_delta_logits": flow_delta[i].tolist(),
                "gated_state_contributions": state_contrib[i].tolist(),
                "calibration_delta": calibration_delta[i].tolist(),
                "identity_check_max_abs": float((final_logits[i] - (global_logits[i] + flow_delta[i] + calibration_delta[i])).abs().max()),
                "tensor_lineage": "final_logits = global_logits + flow_delta_logits + calibration_delta",
            }
            write_json(case_dir / "decision_ledger.json", case)
            _draw_contribution_png(case_dir / "decision_ledger.png", final_logits[i], global_logits[i], flow_delta[i], calibration_delta[i])
            render_case(case, case_dir)
            cases.append({"case_dir": str(case_dir), "file_name": case["file_name"], "pred_action": case["pred_action"], "gt_action": case["gt_action"]})

    rows = "\n".join(
        f"<li><a href='{html.escape(Path(c['case_dir']).name)}/decision_ledger.json'>{html.escape(str(c['file_name']))}</a> "
        f"gt={c['gt_action']} pred={c['pred_action']} "
        f"<br><img src='{html.escape(Path(c['case_dir']).name)}/decision_ledger.png' width='720'></li>"
        for c in cases
    )
    report = f"<html><body><h1>ACPR-InteractFlow++ Decision Ledger</h1><p>available={available}</p><ul>{rows}</ul></body></html>"
    (out / "report.html").write_text(report, encoding="utf-8")
    write_json(
        out / "visual_export_manifest.json",
        {
            "metrics": str(metrics_path),
            "available": available,
            "required_artifacts": required,
            "case_count": len(cases),
            "manual_boxes_forbidden": True,
            "fabricated_effects_forbidden": True,
            "outputs": ["case_*/decision_ledger.json", "case_*/decision_ledger.png", "report.html"],
        },
    )
    if not available:
        raise SystemExit("Missing eval tensors; run eval/train before visual export.")


if __name__ == "__main__":
    main()
