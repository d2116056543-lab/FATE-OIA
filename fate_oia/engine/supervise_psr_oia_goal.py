from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.engine.eval_psr_oia import run_goal
from fate_oia.utils.psr_artifacts import append_jsonl
from fate_oia.utils.psr_review_gates import require_review_pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground PSR-OIA V2 goal supervisor.")
    ap.add_argument("--registry_config", required=True)
    ap.add_argument("--router_config", required=True)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--review_pass_path", default=".background_runs/psr_oia_v2_preflight/REVIEW_PASS_PSR_OIA_V2.txt")
    args = ap.parse_args()
    if args.require_review_pass:
        require_review_pass(args.review_pass_path)
    out_hint = Path(args.output_dir) if args.output_dir else None
    if out_hint:
        append_jsonl(out_hint / "supervisor_decisions.jsonl", {"stage": "start", "decision": "foreground_supervisor_attached"})
    result = run_goal(args.registry_config, args.router_config, args.output_dir or None, args.device)
    print(f"PSR_GOAL_COMPLETED output_dir={result['output_dir']} selected={result['final']['selected']} metrics={result['final']['metrics']}")


if __name__ == "__main__":
    main()
