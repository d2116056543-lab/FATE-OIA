from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from fate_oia.engine.dice_common import build_dice_model, load_config, tensor_state_hash
from fate_oia.utils.dice_artifacts import sha256, write_json
from fate_oia.utils.dice_contracts import assert_base_frozen, assert_probe_contract


REQUIRED = [
    "fate_oia/models/dice_atom_reconstructor.py", "fate_oia/models/dice_directional_head.py",
    "fate_oia/models/dice_license_predictor.py", "fate_oia/models/dice_oia_model.py",
    "fate_oia/losses/dice_rank_sketch.py", "fate_oia/losses/dice_losses.py",
    "fate_oia/losses/dice_loss_registry.py", "fate_oia/utils/dice_counterfactual.py",
    "fate_oia/utils/dice_counterfactual_engine.py",
    "fate_oia/utils/dice_artifacts.py", "fate_oia/utils/dice_metrics.py", "fate_oia/utils/dice_contracts.py",
    "fate_oia/engine/train_dice_oia_probe.py", "fate_oia/engine/evaluate_dice_oia_probe.py",
    "fate_oia/engine/diagnose_dice_oracle.py", "fate_oia/engine/profile_dice_oia.py",
    "fate_oia/engine/supervise_dice_oia_probe.py", "scripts/FATE_OIA_dice_oia_v1_probe.ps1",
]
FORBIDDEN = ("feature_cache_enabled: true", "token_compression: keep_merge", "HardPair", "pair_memory", "reason_logits_final = reason_logits_base +")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--base-checkpoint",required=True)
    parser.add_argument("--output-dir",required=True); parser.add_argument("--device",default="cuda")
    args=parser.parse_args(); root=Path.cwd(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); cfg=load_config(args.config)
    missing=[name for name in REQUIRED if not (root/name).is_file()]
    forbidden={pattern:[] for pattern in FORBIDDEN}
    for path in [*root.glob("fate_oia/**/*dice*.py"),Path(args.config)]:
        if path.name == "audit_dice_oia_probe.py":
            continue
        text=path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if pattern in text: forbidden[pattern].append(str(path))
    checks={"source_head_locked":cfg["experiment"]["source_head"]=="8372dbb0bf0544ad0a3e3b741dc5d3abaab5a5cf",
            "direct_image":cfg["experiment"]["direct_image"] is True,"no_cache":cfg["experiment"]["feature_cache_enabled"] is False,
            "no_compression":cfg["experiment"]["token_compression"]=="none","test_only":cfg["experiment"]["best_selection_split"]=="test",
            "all_required_files":not missing,"forbidden_clean":not any(forbidden.values())}
    error=None; nonfinite_gradients=[]
    try:
        device=torch.device(args.device); model=build_dice_model(cfg,args.base_checkpoint,device); assert_base_frozen(model); before=tensor_state_hash(model.base_model)
        model.train(); image=torch.randn(1,3,360,640,device=device)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            output=model(image); assert_probe_contract(output); loss=output["action_logits_final"].sum()
        loss.backward(); after=tensor_state_hash(model.base_model)
        trainable_gradients=[(name,parameter.grad) for name,parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None]
        nonfinite_gradients=[name for name,gradient in trainable_gradients if not bool(torch.isfinite(gradient).all())]
        checks.update({"dynamic_forward":True,"reason_exact_identity":float(output["reason_identity_max_abs"])==0,
                       "base_hash_unchanged":before==after,"base_grad_zero":all(p.grad is None for p in model.base_model.parameters()),
                       "dice_grad_finite_nonzero":not nonfinite_gradients and any(gradient.abs().sum()>0 for _,gradient in trainable_gradients),
                       "top2_sparse":int(output["predicate_top2_count"].max())<=2,"action_cap":float(output["dice_action_delta"].abs().max())<=.250001})
    except Exception as exc:
        error=repr(exc); checks["dynamic_forward"]=False
    passed=all(checks.values()) and error is None
    payload={"pass":passed,"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             "config_hash":sha256(args.config),"checkpoint_hash":sha256(args.base_checkpoint),"checked_files":REQUIRED,
             "missing_items":missing,"forbidden_pattern_results":forbidden,"functional_checks":checks,
             "nonfinite_gradient_names":nonfinite_gradients,"error":error}
    write_json(out/"DICE_IMPLEMENTATION_REVIEW.json",payload); print(json.dumps(payload,indent=2))
    if not passed: raise SystemExit(1)


if __name__=="__main__": main()
