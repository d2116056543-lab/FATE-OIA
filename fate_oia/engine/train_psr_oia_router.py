from __future__ import annotations

import argparse

from fate_oia.engine.eval_psr_oia import run_goal


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate the logit-level learned PSR router.")
    ap.add_argument("--registry_config", required=True)
    ap.add_argument("--router_config", required=True)
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_goal(args.registry_config, args.router_config, args.output_dir or None, args.device)


if __name__ == "__main__":
    main()
