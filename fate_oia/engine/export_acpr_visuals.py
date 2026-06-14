from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_oia.utils.acpr_artifacts import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--file_name", default="case.jpg")
    ap.add_argument("--matched_file_name", default="")
    ap.add_argument("--predicate_probs_json", default="")
    ap.add_argument("--matched_predicate_probs_json", default="")
    ap.add_argument("--reason", default="unknown")
    ap.add_argument("--reason_logit", type=float, default=0.0)
    ap.add_argument("--matched_reason_logit", type=float, default=0.0)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    current_predicates = json.loads(args.predicate_probs_json) if args.predicate_probs_json else {}
    matched_predicates = json.loads(args.matched_predicate_probs_json) if args.matched_predicate_probs_json else {}
    predicate_delta = {
        k: float(current_predicates.get(k, 0.0)) - float(matched_predicates.get(k, 0.0))
        for k in sorted(set(current_predicates) | set(matched_predicates))
    }
    reason_margin = float(args.reason_logit - args.matched_reason_logit)
    case = {
        "reason": args.reason,
        "current": {"file": args.file_name, "predicate_probs": current_predicates, "reason_logit": args.reason_logit},
        "matched_negative": {"file": args.matched_file_name, "predicate_probs": matched_predicates, "reason_logit": args.matched_reason_logit},
        "predicate_delta": predicate_delta,
        "reason_margin": reason_margin,
        "available": bool(args.matched_file_name),
    }
    write_json(out / "matched_counterfactual_comparison.json", case)
    write_json(out / "predicate_delta.json", predicate_delta)
    write_json(out / "predicate_attention_summary.json", {"available": bool(current_predicates), "predicate_delta": predicate_delta})
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ACPR matched counterfactual report</title></head>
<body>
<h1>ACPR matched counterfactual comparison</h1>
<p>Reason: {args.reason}</p>
<p>Current image: {args.file_name}</p>
<p>Matched negative: {args.matched_file_name}</p>
<p>Reason margin: {reason_margin:.6f}</p>
<pre>{json.dumps(case, ensure_ascii=False, indent=2)}</pre>
</body></html>
"""
    (out / "report.html").write_text(html, encoding="utf-8")
    write_json(out / "report.json", {"available": True, "html": "report.html", "reason_margin": reason_margin})


if __name__ == "__main__":
    main()
