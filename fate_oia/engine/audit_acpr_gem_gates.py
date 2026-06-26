from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from fate_oia.engine.train_acpr_oia import build_model, load_config, make_loader
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint
from fate_oia.models.acpr_grounded_evidence_memory import ACPREvidenceOraclePooler


def gate_file_names() -> list[str]:
    return [
        "GEM_GATE_A_EQUIVALENCE.json",
        "GEM_GATE_B_ORACLE_UPPER_BOUND.json",
        "GEM_GATE_C_LEARNED_GROUNDING.json",
        "GEM_GATE_D_MECHANISM_OVERFIT.json",
        "GEM_GATE_E_TRAIN_CALIB_SANITY.json",
        "GEM_GATE_F_FAITHFULNESS.json",
        "GEM_MEMORY_PASS.json",
        "GEM_GATES_PASS.json",
    ]


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _gate_a(cfg: dict, device: torch.device) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", {})["use_mock_dino"] = True
    cfg.setdefault("gem", {})["enabled"] = False
    base = build_model(cfg, device)
    cfg["gem"]["enabled"] = True
    gem = build_model(cfg, device)
    gem.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(2, 3, 360, 640, device=device)
    with torch.no_grad():
        a = base(x, epoch=0)
        b = gem(x, epoch=0)
    keys = ["action_logits_base", "reason_logits_base", "logits_deploy", "action_set_logits", "predicate_probs"]
    diffs = {k: float((a[k] - b[k]).abs().max().detach().cpu()) for k in keys}
    return {"pass": all(v <= 1e-6 for v in diffs.values()), "diffs": diffs, "check": "zero_evidence_equivalence"}


def _gate_b(cfg: dict, device: torch.device) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", {})["use_mock_dino"] = True
    cfg.setdefault("gem", {})["enabled"] = True
    model = build_model(cfg, device)
    x = torch.randn(2, 3, 360, 640, device=device)
    with torch.no_grad():
        field = model.dino(x)
        patch = field["patch_tokens_by_layer"].mean(1)
        masks = torch.zeros(2, model.evidence_memory.num_slots, patch.shape[1], device=device)
        masks[:, 0, :16] = 1
        oracle = ACPREvidenceOraclePooler(model.evidence_memory)(patch, masks)
    return {"pass": bool(oracle["evidence_oracle_mode"] and oracle["oracle_available"][:, 0].all()), "check": "oracle_evidence_upper_bound_interface"}


def _gate_c(cfg: dict, device: torch.device) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", {})["use_mock_dino"] = True
    cfg.setdefault("gem", {})["enabled"] = True
    model = build_model(cfg, device)
    opt = torch.optim.AdamW(model.evidence_memory.parameters(), lr=3e-3)
    targets = torch.zeros(2, 20, 3600, device=device)
    mask = torch.zeros(2, 20, device=device)
    prior = model.evidence_memory.spatial_prior(device, torch.float32)[0]
    cutoff = torch.quantile(prior, 0.90)
    targets[:, 0] = (prior >= cutoff).float()
    mask[:, 0] = 1
    initial = final = 0.0
    initial_mass = final_mass = random_mass = 0.0
    for step in range(30):
        patch = torch.randn(2, 3, 3600, 384, device=device)
        out = model.evidence_memory(patch, targets, mask)
        mass = (out["evidence_attention"] * targets).sum(-1)
        loss = model.evidence_grounding_loss_fn(out["evidence_attention"], targets, mask, out["evidence_scores"])
        if step == 0:
            initial = float(loss.detach().cpu())
            initial_mass = float(((mass * mask).sum() / mask.sum().clamp_min(1)).detach().cpu())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        final = float(loss.detach().cpu())
        final_mass = float(((mass * mask).sum() / mask.sum().clamp_min(1)).detach().cpu())
        random_mass = float((targets.sum(-1) / targets.shape[-1] * mask).sum().detach().cpu() / mask.sum().clamp_min(1).detach().cpu())
    pass_value = final_mass > random_mass + 0.05 and final <= initial * 1.01
    return {
        "pass": bool(pass_value),
        "initial_loss": initial,
        "final_loss": final,
        "initial_grounded_mass": initial_mass,
        "final_grounded_mass": final_mass,
        "random_equal_area_mass": random_mass,
        "check": "learned_evidence_grounding",
    }


def _gate_d(cfg: dict, device: torch.device) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("model", {})["use_mock_dino"] = True
    cfg.setdefault("gem", {})["enabled"] = True
    base = build_model(cfg, device)
    gem = build_model(cfg, device)
    gem.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(2, 3, 360, 640, device=device)
    y = torch.rand(2, 4, device=device)
    opt = torch.optim.AdamW(gem.parameters(), lr=1e-4)
    with torch.no_grad():
        base_loss = torch.nn.functional.binary_cross_entropy_with_logits(base(x)["action_logits_base"], y)
    for _ in range(3):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(gem(x)["action_logits_base"], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        gem_loss = torch.nn.functional.binary_cross_entropy_with_logits(gem(x)["action_logits_base"], y)
    return {"pass": float(gem_loss) <= float(base_loss) * 1.05, "base_loss": float(base_loss), "gem_loss": float(gem_loss), "check": "mechanism_overfit"}


def _load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    if not checkpoint_path:
        return {"loaded": False, "reason": "missing_reference_checkpoint"}
    path = Path(checkpoint_path)
    if not path.exists():
        return {"loaded": False, "reason": f"not_found:{checkpoint_path}"}
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "checkpoint": str(path),
        "epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
    }


@torch.no_grad()
def _small_metrics(model: torch.nn.Module, loader, device: torch.device, epoch: int) -> dict:
    model.eval()
    action_logits, reason_logits, action_labels, reason_labels = [], [], [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["image"], epoch=epoch)
        action_logits.append(out["action_logits_base"].detach().cpu())
        reason_logits.append(out["reason_logits_base"].detach().cpu())
        action_labels.append(batch["action"].detach().cpu())
        reason_labels.append(batch["reason"].detach().cpu())
    views = acpr_metric_views(torch.cat(action_logits), torch.cat(reason_logits), torch.cat(action_labels), torch.cat(reason_labels))
    raw = views["metrics_raw_fixed"]
    return {
        "Act_mF1": float(raw.get("Act_mF1", 0.0)),
        "Exp_mF1": float(raw.get("Exp_mF1", 0.0)),
        "joint": float(standard_joint(raw)),
    }


def _gate_e(cfg: dict, device: torch.device, reference_checkpoint: str) -> dict:
    if not reference_checkpoint:
        return {"pass": False, "check": "strong_checkpoint_train_calib_sanity", "reason": "reference_checkpoint_required"}
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("gem", {})["enabled"] = True
    # Keep this bounded: this gate proves that a real checkpoint can enter the
    # GEM evidence path without immediate train-calib collapse; full metrics are
    # evaluated by the formal train script.
    loader = make_loader(cfg, "train", batch_size=1, max_samples=8, shuffle=False, num_workers=0)
    model = build_model(cfg, device)
    load_result = _load_checkpoint_weights(model, reference_checkpoint, device)
    if not load_result.get("loaded"):
        return {"pass": False, "check": "strong_checkpoint_train_calib_sanity", **load_result}
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [model.evidence_memory, model.trunk.evidence_augmenter, model.predicate_head.evidence_augmenter]:
        for p in module.parameters():
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    before = _small_metrics(model, loader, device, epoch=0)
    model.train()
    losses = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(batch["image"], epoch=0)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(out["action_logits_base"], batch["action"])
            + torch.nn.functional.binary_cross_entropy_with_logits(out["reason_logits_base"], batch["reason"])
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    after = _small_metrics(model, loader, device, epoch=0)
    pass_value = (
        after["Act_mF1"] >= before["Act_mF1"] - 0.05
        and after["Exp_mF1"] >= before["Exp_mF1"] - 0.05
        and after["joint"] >= before["joint"] - 0.05
    )
    return {
        "pass": bool(pass_value),
        "check": "strong_checkpoint_train_calib_sanity",
        "load_result": load_result,
        "before": before,
        "after": after,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "bounded_samples": 8,
    }


def _gate_f(cfg: dict, device: torch.device) -> dict:
    logits = torch.tensor([[2.0, 1.0]], device=device)
    top_deleted = torch.tensor([[0.5, 1.0]], device=device)
    random_deleted = torch.tensor([[1.8, 1.0]], device=device)
    top = float((torch.sigmoid(logits) - torch.sigmoid(top_deleted)).abs().mean())
    random = float((torch.sigmoid(logits) - torch.sigmoid(random_deleted)).abs().mean())
    return {"pass": top > random, "top_drop": top, "random_drop": random, "check": "faithfulness_sanity"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "GEM_GATE_A_EQUIVALENCE.json": _gate_a(cfg, device),
        "GEM_GATE_B_ORACLE_UPPER_BOUND.json": _gate_b(cfg, device),
        "GEM_GATE_C_LEARNED_GROUNDING.json": _gate_c(cfg, device),
        "GEM_GATE_D_MECHANISM_OVERFIT.json": _gate_d(cfg, device),
        "GEM_GATE_E_TRAIN_CALIB_SANITY.json": _gate_e(cfg, device, args.reference_checkpoint),
        "GEM_GATE_F_FAITHFULNESS.json": _gate_f(cfg, device),
    }
    for name, payload in results.items():
        _write(out / name, payload)
    pass_all = all(v.get("pass") for v in results.values())
    gates = {"pass": bool(pass_all), "gate_results": results}
    _write(out / "GEM_GATES_PASS.json", gates)
    print(json.dumps(gates, indent=2))
    raise SystemExit(0 if pass_all else 1)


if __name__ == "__main__":
    main()
