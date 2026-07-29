from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.losses.meter_action_losses import meter_action_loss
from fate_oia.losses.meter_grounding_losses import meter_grounding_loss
from fate_oia.losses.meter_pu_losses import meter_private_pu_loss
from fate_oia.losses.meter_reason_losses import meter_reason_loss
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.utils.meter_artifacts import write_json
from fate_oia.utils.meter_config import load_meter_config


REQUIRED_FILES = (
    "fate_oia/models/meter_signed_factors.py",
    "fate_oia/models/meter_semantic_action.py",
    "fate_oia/models/meter_reason_decoder.py",
    "fate_oia/models/meter_oia_model.py",
    "fate_oia/losses/meter_grounding_losses.py",
    "fate_oia/losses/meter_counterfactual_losses.py",
    "fate_oia/losses/meter_pu_losses.py",
    "fate_oia/engine/train_acpr_meter_oia.py",
    "fate_oia/engine/eval_acpr_meter_oia.py",
    "fate_oia/engine/tesa_diagnostics.py",
    "configs/meter_factor_schema.yaml",
)
FORBIDDEN_FORMAL = (
    "action_selector",
    "selector_regret",
    "reason_logits_local",
    "reason_mix_gate",
    "reason_annotation_delta",
    "_counterfactual_event",
    "formal_meta_event",
    "factor_support_map",
    "factor_counter_map",
)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unavailable"


def _source_checks(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    formal_paths = (
        root / "fate_oia/models/meter_oia_model.py",
        root / "fate_oia/engine/train_acpr_meter_oia.py",
        root / "fate_oia/engine/eval_acpr_meter_oia.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in formal_paths)
    forbidden = {token: token in text for token in FORBIDDEN_FORMAL}
    schema = yaml.safe_load(
        (root / "configs/meter_factor_schema.yaml").read_text(encoding="utf-8")
    )
    rows = schema.get("factors", [])
    schema_ok = (
        len(rows) == 21
        and [int(row["id"]) for row in rows] == list(range(21))
        and all(
            {
                "factor_type",
                "state_set",
                "anchor_source",
                "state_source",
                "groundability",
                "action_owned",
                "observability_source",
                "mirror_partner",
                "counter_localizable",
            }
            <= set(row)
            for row in rows
        )
        and all(row["counter_localizable"] is False for row in rows)
    )
    protocol = {
        "direct_image": bool(config["experiment"]["direct_image"]),
        "no_cache": config["model"]["feature_cache_enabled"] is False,
        "no_compression": config["model"]["token_compression"] == "none",
        "test_only": config["runtime"]["test_only"] is True,
        "meta_audit_only": config["meta"]["training_enabled"] is False
        and config["meta"]["audit_only"] is True,
        "sequential_eval": config["runtime"]["sequential_eval"] is True,
    }
    return {
        "missing_files": missing,
        "forbidden_patterns": forbidden,
        "schema_ok": schema_ok,
        "protocol": protocol,
        "pass": not missing
        and not any(forbidden.values())
        and schema_ok
        and all(protocol.values()),
    }


def _gradient_norm(module: torch.nn.Module) -> float:
    return float(
        sum(
            parameter.grad.detach().float().norm().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def _dynamic_checks(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(20260729)
    model = METEROIAModel(dim=32, use_mock_dino=True).to(device)
    images = torch.randn(2, 3, 360, 640, device=device)
    action_target = torch.randint(0, 2, (2, 4), device=device).float()
    reason_target = torch.randint(0, 2, (2, 21), device=device).float()
    progress_zero = model(images, progress=0.0)
    progress_one = model(images, progress=1.0)
    shapes = {
        "action_logits_final": list(progress_one["action_logits_final"].shape),
        "reason_logits_final": list(progress_one["reason_logits_final"].shape),
        "factor_anchor_map": list(progress_one["factor_anchor_map"].shape),
        "factor_state_prob": list(progress_one["factor_state_prob"].shape),
        "action_factor_contributions": list(
            progress_one["action_factor_contributions"].shape
        ),
    }
    additive_error = float(
        (
            progress_one["action_logits_final"]
            - progress_one["action_logits_visual"]
            - torch.tanh(
                progress_one["action_factor_contributions"].sum(-1)
                / progress_one["action_correction_kappa"].view(1, -1)
            )
            * progress_one["action_correction_kappa"].view(1, -1)
        )
        .abs()
        .max()
    )
    zero_action_error = float(
        (
            progress_zero["action_logits_final"]
            - progress_zero["action_logits_calalign"]
        )
        .abs()
        .max()
    )
    zero_reason_error = float(
        (
            progress_zero["reason_logits_global"]
            - progress_zero["reason_logits_calalign"]
        )
        .abs()
        .max()
    )
    model.zero_grad(set_to_none=True)
    meter_action_loss(progress_one, action_target)["total"].backward(retain_graph=True)
    action_grads = {
        "foundation": _gradient_norm(model.foundation),
        "factor": _gradient_norm(model.typed_factors),
        "action": _gradient_norm(model.action_transport),
        "reason": _gradient_norm(model.reason_decoder),
    }
    model.zero_grad(set_to_none=True)
    meter_reason_loss(
        progress_one,
        reason_target,
        progress_one["factor_reliability"].detach(),
        observability=progress_one["factor_observability"].detach(),
    )["total"].backward(retain_graph=True)
    reason_grads = {
        "foundation": _gradient_norm(model.foundation),
        "factor": _gradient_norm(model.typed_factors),
        "action": _gradient_norm(model.action_transport),
        "reason": _gradient_norm(model.reason_decoder),
    }
    model.zero_grad(set_to_none=True)
    zero_lambda = torch.zeros(21, device=device)
    pu = meter_private_pu_loss(
        progress_one["reason_logits_pu_private"],
        reason_target,
        torch.rand_like(reason_target),
        zero_lambda,
    )
    pu.backward()
    pu_zero = float(pu.detach()) == 0.0 and all(
        parameter.grad is None or float(parameter.grad.abs().max()) == 0.0
        for parameter in model.parameters()
    )
    finite = all(
        bool(torch.isfinite(progress_one[key]).all())
        for key in (
            "action_logits_final",
            "reason_logits_final",
            "factor_anchor_map",
            "factor_state_prob",
            "factor_observability",
            "factor_reliability",
        )
    )
    return {
        "shapes": shapes,
        "progress_zero_action_error": zero_action_error,
        "progress_zero_reason_error": zero_reason_error,
        "additive_error": additive_error,
        "action_gradient_ownership": action_grads,
        "reason_gradient_firewall": reason_grads,
        "pu_zero_exact": pu_zero,
        "finite": finite,
        "pass": (
            shapes["action_logits_final"] == [2, 4]
            and shapes["reason_logits_final"] == [2, 21]
            and shapes["factor_anchor_map"] == [2, 21, 3600]
            and shapes["factor_state_prob"] == [2, 21, 3]
            and shapes["action_factor_contributions"] == [2, 4, 21]
            and zero_action_error < 1e-6
            and zero_reason_error < 1e-6
            and additive_error < 1e-5
            and action_grads["reason"] == 0.0
            and reason_grads["foundation"] == 0.0
            and reason_grads["factor"] == 0.0
            and reason_grads["action"] == 0.0
            and reason_grads["reason"] > 0.0
            and pu_zero
            and finite
        ),
    }


def run_audit(
    config_path: str,
    output_dir: str,
    *,
    device: str,
    write_review_pass: bool,
) -> dict[str, Any]:
    root = Path.cwd()
    config = load_meter_config(config_path)
    compile_errors: list[str] = []
    for name in REQUIRED_FILES:
        path = root / name
        if path.suffix == ".py" and path.exists():
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as error:
                compile_errors.append(f"{name}: {error}")
    source = _source_checks(root, config)
    dynamic = _dynamic_checks(torch.device(device))
    result = {
        "pass": not compile_errors and source["pass"] and dynamic["pass"],
        "git_head": _git_head(),
        "compile_errors": compile_errors,
        "source_checks": source,
        "dynamic_checks": dynamic,
        "pilot_gates": {
            "required_before_full_train": True,
            "status": "not_evaluated_by_implementation_audit",
        },
        "review_pass_path": "",
        "missing_items": source["missing_files"],
        "warnings": [],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "implementation_audit_METER_OIA_V2_TESA.json"
    if write_review_pass and result["pass"]:
        review = output / "REVIEW_PASS_METER_OIA_V2_TESA.txt"
        review.write_text(
            json.dumps(
                {
                    "git_head": result["git_head"],
                    "implementation_audit": str(audit_path),
                    "pilot_gates_required": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        result["review_pass_path"] = str(review)
    write_json(audit_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        args.config,
        args.output_dir,
        device=args.device,
        write_review_pass=args.write_review_pass,
    )
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
