from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from fate_oia.models.precise_pcvl_probes import PRECISEPCVLProbes


PCVL_FILES = ("pcvl_metrics.json", "pcvl_per_action.json", "pcvl_bootstrap.json", "pcvl_value_decomposition.json")


def build_oracle_structured_evidence(
    structured_targets: dict[str, torch.Tensor], learned_evidence: torch.Tensor
) -> torch.Tensor:
    """Build pilot-only oracle evidence from train grounding targets."""
    presence = structured_targets["presence"].to(learned_evidence).float()
    valid = structured_targets["presence_valid"].to(learned_evidence).float()
    if presence.shape != learned_evidence.shape[:2] or valid.shape != presence.shape:
        raise ValueError("Oracle evidence targets do not match explicit evidence fields")
    observability = structured_targets["observability"].to(learned_evidence).float() * valid
    state_valid = structured_targets.get("state_valid", valid).to(learned_evidence).float()
    state = structured_targets["state"].to(learned_evidence).float() * state_valid.unsqueeze(-1)
    geometry_valid = structured_targets.get("part_valid", presence * valid).to(learned_evidence).float() * presence * valid
    coordinates = structured_targets["part_coordinates"].to(learned_evidence).float().mean(-2) * geometry_valid.unsqueeze(-1)
    scales = structured_targets.get("part_scales", torch.zeros_like(structured_targets["part_coordinates"])).to(learned_evidence).float().mean(-2) * geometry_valid.unsqueeze(-1)
    masks = structured_targets.get("soft_masks")
    if masks is None:
        mask_stats = learned_evidence.new_zeros(*presence.shape, 3)
    else:
        masks = masks.to(learned_evidence).float()
        height, width = masks.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, height, device=masks.device), torch.linspace(0.0, 1.0, width, device=masks.device), indexing="ij")
        mass = masks.sum((-1, -2)).clamp_min(1e-6)
        mask_stats = torch.stack([masks.mean((-1, -2)), (masks * xx).sum((-1, -2)) / mass, (masks * yy).sum((-1, -2)) / mass], dim=-1) * geometry_valid.unsqueeze(-1)
    descriptor = torch.cat([((2.0 * presence - 1.0) * valid).unsqueeze(-1), observability.unsqueeze(-1), state, coordinates, scales, mask_stats], dim=-1)
    repeats = (learned_evidence.shape[-1] + descriptor.shape[-1] - 1) // descriptor.shape[-1]
    return descriptor.repeat(1, 1, repeats)[..., : learned_evidence.shape[-1]].detach()


def pcvl_inputs(output: dict[str, torch.Tensor], structured_targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    learned = output["explicit_evidence_tokens"].detach()
    oracle = build_oracle_structured_evidence(structured_targets, learned)
    compatibility = output["action_evidence_family_mask"].to(learned).float()
    compatibility = compatibility / compatibility.sum(-1, keepdim=True).clamp_min(1.0)
    oracle_by_action = torch.einsum("ae,bed->bad", compatibility, oracle)
    learned_by_action = torch.einsum("ae,bed->bad", compatibility, learned)
    return output["action_tokens_direct"].detach(), oracle_by_action, learned_by_action, output["action_exchange_delta"].detach()


def train_pcvl_step(
    probes: PRECISEPCVLProbes,
    optimizer: torch.optim.Optimizer,
    output: dict[str, torch.Tensor],
    structured_targets: dict[str, torch.Tensor],
    action_targets: torch.Tensor,
) -> float:
    logits = probes(*pcvl_inputs(output, structured_targets))
    loss = sum(F.binary_cross_entropy_with_logits(value, action_targets.float()) for value in logits.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _average_precision(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    values = []
    for label in range(targets.shape[1]):
        order = probabilities[:, label].argsort(descending=True)
        truth = targets[order, label].float()
        positives = truth.sum()
        if positives == 0:
            values.append(torch.tensor(float("nan"), device=targets.device))
            continue
        precision = truth.cumsum(0) / torch.arange(1, len(truth) + 1, device=truth.device)
        values.append((precision * truth).sum() / positives)
    return torch.stack(values)


@torch.no_grad()
def evaluate_pcvl(
    probes: PRECISEPCVLProbes,
    model,
    loader,
    target_provider,
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    probes.eval()
    model.eval()
    predictions = {key: [] for key in ("u0", "u1", "u2", "u3")}
    labels = []
    for batch in loader:
        output = model(batch["image"].to(device, non_blocking=True))
        structured = target_provider(batch, device)
        logits = probes(*pcvl_inputs(output, structured))
        for key, value in logits.items():
            predictions[key].append(torch.sigmoid(value).cpu())
        labels.append(batch["action"].cpu())
    target = torch.cat(labels)
    per_action = {}
    metrics = {}
    for key, chunks in predictions.items():
        ap = _average_precision(torch.cat(chunks), target)
        per_action[key] = [None if torch.isnan(value) else float(value) for value in ap]
        metrics[f"{key}_action_map"] = float(torch.nanmean(ap))
    decomposition = {
        "delta_value": metrics["u1_action_map"] - metrics["u0_action_map"],
        "delta_measurement": metrics["u2_action_map"] - metrics["u1_action_map"],
        "delta_interaction": metrics["u3_action_map"] - metrics["u2_action_map"],
    }
    metrics["predicate_action_value_supported"] = bool(decomposition["delta_value"] > 0.0)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pcvl_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (root / "pcvl_per_action.json").write_text(json.dumps(per_action, indent=2), encoding="utf-8")
    probability_tensors = {key: torch.cat(chunks) for key, chunks in predictions.items()}
    generator = torch.Generator().manual_seed(20260722)
    bootstrap_rows = []
    for _ in range(200):
        sample = torch.randint(0, len(target), (len(target),), generator=generator)
        sampled_target = target[sample]
        sampled_map = {key: float(torch.nanmean(_average_precision(value[sample], sampled_target))) for key, value in probability_tensors.items()}
        bootstrap_rows.append({
            "delta_value": sampled_map["u1"] - sampled_map["u0"],
            "delta_measurement": sampled_map["u2"] - sampled_map["u1"],
            "delta_interaction": sampled_map["u3"] - sampled_map["u2"],
        })
    bootstrap = {}
    for name in ("delta_value", "delta_measurement", "delta_interaction"):
        values = torch.tensor([row[name] for row in bootstrap_rows])
        bootstrap[name] = {
            "mean": float(values.mean()),
            "ci_low": float(values.quantile(0.025)),
            "ci_high": float(values.quantile(0.975)),
            "positive_rate": float((values > 0).float().mean()),
        }
    (root / "pcvl_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    (root / "pcvl_value_decomposition.json").write_text(json.dumps(decomposition, indent=2), encoding="utf-8")
    probes.train()
    return {**metrics, **decomposition}


def validate_pcvl_artifacts(path: str | Path) -> None:
    root = Path(path)
    missing = [name for name in PCVL_FILES if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"Missing PCVL pilot artifacts: {missing}")
    metrics = json.loads((root / "pcvl_metrics.json").read_text(encoding="utf-8"))
    for key in ("u0_action_map", "u1_action_map", "u2_action_map", "u3_action_map", "predicate_action_value_supported"):
        if key not in metrics:
            raise RuntimeError(f"PCVL metrics missing {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate_dir", required=True)
    args = parser.parse_args()
    validate_pcvl_artifacts(args.validate_dir)


if __name__ == "__main__":
    main()
