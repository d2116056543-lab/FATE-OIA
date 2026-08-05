from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from fate_oia.models.lens_oia_model import LENSOIAModel
from fate_oia.utils.lens_artifacts import write_json
from fate_oia.utils.lens_contracts import assert_lens_forward_contract
from fate_oia.losses.lens_latent_losses import conflict_safe_reason_logits
from fate_oia.engine.train_lens_oia import mechanism_progress, should_optimizer_step


REQUIRED = [
    "fate_oia/models/lens_calalign_foundation.py", "fate_oia/models/lens_adaptive_evidence.py", "fate_oia/models/lens_latent_state.py",
    "fate_oia/models/lens_annotation_emission.py", "fate_oia/models/lens_action_reread.py", "fate_oia/models/lens_oia_model.py",
    "fate_oia/datasets/lens_structured_evidence.py", "fate_oia/datasets/lens_splits.py", "fate_oia/datasets/lens_mirror.py",
    "fate_oia/losses/lens_action_losses.py", "fate_oia/losses/lens_reason_losses.py", "fate_oia/losses/lens_latent_losses.py", "fate_oia/losses/lens_grounding_losses.py", "fate_oia/losses/lens_loss_registry.py",
    "fate_oia/engine/train_lens_oia.py", "fate_oia/engine/eval_lens_oia.py", "fate_oia/engine/profile_lens_oia.py", "fate_oia/engine/evaluate_lens_oia_pilot.py", "fate_oia/engine/supervise_lens_oia_foreground.py",
    "fate_oia/utils/lens_artifacts.py", "fate_oia/utils/lens_calibration.py", "fate_oia/utils/lens_metrics.py", "fate_oia/utils/lens_contracts.py", "fate_oia/utils/lens_hashes.py",
    "scripts/FATE_OIA_lens_oia_v1_pilot.ps1", "scripts/FATE_OIA_lens_oia_v1_foreground.ps1",
]
FORBIDDEN = ("ACPRPairMemory", "matched_pair", "pair_memory", "ACPRActionComboAux", "ACPRThresholdHead", "SAVEUtilityBridge", "PCGrad", "cached_logits", "FrozenRunC", "token_compression: keep_merge")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--device",default="cuda"); parser.add_argument("--allow-mock-dino",action="store_true"); args=parser.parse_args()
    root=Path.cwd(); config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    missing=[path for path in REQUIRED if not (root/path).exists()]
    audited_roots = (root / "fate_oia" / "models", root / "fate_oia" / "losses", root / "fate_oia" / "datasets", root / "fate_oia" / "utils")
    audited_files = [path for directory in audited_roots for path in directory.glob("lens_*.py")]
    audited_files.append(root / "fate_oia" / "engine" / "train_lens_oia.py")
    lens_sources="\n".join(path.read_text(encoding="utf-8") for path in audited_files if path.exists())
    forbidden=[token for token in FORBIDDEN if token in lens_sources]
    functional={}; failures=[]
    try:
        use_mock = bool(args.allow_mock_dino)
        model=LENSOIAModel(use_mock_dino=use_mock, pretrained_weights=str(config["pretrained_weights"])).to(args.device)
        calls = {"count": 0}; original_encode = model.foundation.dino.forward
        def counted_encode(*values, **kwargs):
            calls["count"] += 1
            return original_encode(*values, **kwargs)
        model.foundation.dino.forward = counted_encode  # type: ignore[method-assign]
        out=model(torch.randn(1,3,360,640,device=args.device),progress=0.0); assert_lens_forward_contract(out,1)
        functional["progress_zero_exact"] = bool(torch.equal(out["action_logits_final"],out["action_logits_source"]) and torch.equal(out["reason_logits_formal"],out["reason_logits_source"]))
        functional["no_dino_grad"] = all(not parameter.requires_grad for parameter in model.foundation.dino.parameters())
        functional["emission_ordered"] = bool(torch.all(out["emission_prob"][:,2] > out["emission_prob"][:,1]) and torch.all(out["emission_prob"][:,1] > out["emission_prob"][:,0]))
        functional["contribution_exact"] = float(out["contribution_reconstruction_error"]) < 1e-6
        functional["one_dino_call"] = calls["count"] == 1
        # Formal reason supervision may update shared evidence/state, but never the action rereader nor action-only head.
        model.zero_grad(set_to_none=True); full = model(torch.randn(2,3,360,640,device=args.device),progress=1.0)
        F.binary_cross_entropy_with_logits(full["reason_logits_formal"], torch.randint(0,2,(2,21),device=args.device).float()).backward()
        action_head_grad = sum(float((p.grad.abs().sum() if p.grad is not None else 0.0)) for p in model.foundation.trunk.action_visual_head.parameters())
        reread_grad = sum(float((p.grad.abs().sum() if p.grad is not None else 0.0)) for p in model.action_reread.parameters())
        functional["reason_to_action_firewall"] = action_head_grad == 0.0 and reread_grad == 0.0
        model.zero_grad(set_to_none=True); full = model(torch.randn(2,3,360,640,device=args.device),progress=1.0)
        F.binary_cross_entropy_with_logits(full["action_logits_final"], torch.randint(0,2,(2,4),device=args.device).float()).backward()
        evidence_grad = sum(float((p.grad.abs().sum() if p.grad is not None else 0.0)) for p in model.adaptive_evidence.parameters())
        reread_grad = sum(float((p.grad.abs().sum() if p.grad is not None else 0.0)) for p in model.action_reread.parameters())
        emission_grad = sum(float((p.grad.abs().sum() if p.grad is not None else 0.0)) for p in model.annotation_emission.parameters())
        functional["action_owner_path"] = evidence_grad > 0.0 and reread_grad > 0.0 and emission_grad == 0.0
        state_logits=torch.randn(2,21,3,device=args.device,requires_grad=True); state=state_logits.softmax(-1); source=torch.randn(2,21,device=args.device,requires_grad=True); gamma=torch.softmax(torch.randn(2,21,3,device=args.device),-1); observed=torch.zeros(2,21,device=args.device)
        safe=conflict_safe_reason_logits(state,source,full["emission_prob"],observed,gamma,torch.full((2,21),0.95,device=args.device),1.0)
        safe["reason_logits_formal_train"].sum().backward(); high_conflict_grad=float(state_logits.grad.norm()+source.grad.norm())
        functional["conflict_changes_shared_gradient"] = high_conflict_grad > 0.0 and float(safe["share_weight"].min()) >= 0.05
        functional["accumulation_tail_flush"] = should_optimizer_step(0,1,5) and should_optimizer_step(5,6,5)
        functional["update_based_ramp"] = mechanism_progress(5,100,0.10)==0.5
        functional["state_specific_reread"] = hasattr(model.action_reread,"named_weight") and not torch.allclose(full["factor_contribution_state"][...,0],-full["factor_contribution_state"][...,1])
    except Exception as exc:
        failures.append(repr(exc)); functional["forward"] = False
    protocol = config.get("eval",{}).get("best_selection_split")=="test" and config.get("feature_cache_enabled") is False and config.get("token_compression")=="none"
    real_dino = not bool(args.allow_mock_dino)
    passed=not missing and not forbidden and not failures and all(functional.values()) and protocol and real_dino
    status="REVIEW_PASS" if passed else "REVIEW_FAIL"
    git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    config_hash=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(); schema_path=Path(config.get("reason_state_schema","configs/lens_reason_state_schema.yaml")); schema_hash=hashlib.sha256(schema_path.read_bytes()).hexdigest()
    payload={"pass":passed,"status":status,"git_head":git_head,"config_hash":config_hash,"schema_hash":schema_hash,"split_seed":config.get("splits",{}).get("seed"),"checked_files":REQUIRED,"missing_files":missing,"forbidden_paths":forbidden,"functional_checks":functional,"protocol_ok":protocol,"real_dino":real_dino,"failures":failures,"warnings":[] if real_dino else ["Mock-DINO audit cannot issue REVIEW_PASS."]}
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); write_json(output/"LENS_IMPLEMENTATION_REVIEW.json",payload)
    print(json.dumps(payload,ensure_ascii=False))


if __name__=="__main__": main()
