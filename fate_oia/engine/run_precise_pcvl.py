from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from fate_oia.models.precise_pcvl_probes import PRECISEPCVLProbes


PCVL_FILES = (
    "pcvl_metrics.json", "pcvl_per_action.json", "pcvl_bootstrap.json",
    "pcvl_value_decomposition.json", "pcvl_provenance.json",
    "pcvl_probabilities.pt", "pcvl_labels.pt", "pcvl_file_names.json",
)


def _state_sha256(module: torch.nn.Module, *, trainable_only: bool = False) -> str:
    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if trainable_only and name not in trainable:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


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
    raw_coordinates = structured_targets["part_coordinates"].to(learned_evidence).float()
    raw_scales = structured_targets.get("part_scales", torch.zeros_like(structured_targets["part_coordinates"])).to(learned_evidence).float()
    # Batch collation pads heterogeneous fields to the largest part count.
    # Scale is zero only for padded slots, so it is the stable per-part mask.
    part_mask = raw_scales.abs().sum(-1) > 0
    part_count = part_mask.sum(-1, keepdim=True).clamp_min(1).to(raw_coordinates)
    coordinates = (raw_coordinates * part_mask.unsqueeze(-1)).sum(-2) / part_count
    scales = (raw_scales * part_mask.unsqueeze(-1)).sum(-2) / part_count
    coordinates = coordinates * geometry_valid.unsqueeze(-1)
    scales = scales * geometry_valid.unsqueeze(-1)
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
) -> dict[str, float]:
    logits = probes(*pcvl_inputs(output, structured_targets))
    loss = sum(F.binary_cross_entropy_with_logits(value, action_targets.float()) for value in logits.values())
    optimizer.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in probes.parameters()]
    loss.backward()
    gradients = [parameter.grad.detach().norm() for parameter in probes.parameters() if parameter.grad is not None]
    grad_norm = torch.stack(gradients).norm() if gradients else loss.new_zeros(())
    optimizer.step()
    deltas = [(parameter.detach() - old).norm() for parameter, old in zip(probes.parameters(), before)]
    delta_norm = torch.stack(deltas).norm() if deltas else loss.new_zeros(())
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm), "parameter_delta_norm": float(delta_norm)}


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
    provenance: dict[str, Any],
) -> dict[str, Any]:
    probes.eval()
    model.eval()
    predictions = {key: [] for key in ("u0", "u1", "u2", "u3")}
    labels = []
    file_names: list[str] = []
    for batch in loader:
        output = model(batch["image"].to(device, non_blocking=True))
        structured = target_provider(batch, device)
        logits = probes(*pcvl_inputs(output, structured))
        for key, value in logits.items():
            predictions[key].append(torch.sigmoid(value).cpu())
        labels.append(batch["action"].cpu())
        file_names.extend(str(name) for name in batch["file_name"])
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
        "delta_learned_value": metrics["u2_action_map"] - metrics["u0_action_map"],
        "delta_learned_interaction": metrics["u3_action_map"] - metrics["u2_action_map"],
    }
    metrics["predicate_action_value_supported"] = bool(decomposition["delta_value"] > 0.0)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pcvl_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (root / "pcvl_per_action.json").write_text(json.dumps(per_action, indent=2), encoding="utf-8")
    probability_tensors = {key: torch.cat(chunks) for key, chunks in predictions.items()}
    torch.save(probability_tensors, root / "pcvl_probabilities.pt")
    torch.save(target, root / "pcvl_labels.pt")
    (root / "pcvl_file_names.json").write_text(json.dumps({"file_names": file_names}, indent=2), encoding="utf-8")
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
            "delta_learned_value": sampled_map["u2"] - sampled_map["u0"],
            "delta_learned_interaction": sampled_map["u3"] - sampled_map["u2"],
        })
    bootstrap = {}
    for name in ("delta_value", "delta_measurement", "delta_interaction", "delta_learned_value", "delta_learned_interaction"):
        values = torch.tensor([row[name] for row in bootstrap_rows])
        bootstrap[name] = {
            "mean": float(values.mean()),
            "ci_low": float(values.quantile(0.025)),
            "ci_high": float(values.quantile(0.975)),
            "positive_rate": float((values > 0).float().mean()),
        }
    (root / "pcvl_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    (root / "pcvl_value_decomposition.json").write_text(json.dumps(decomposition, indent=2), encoding="utf-8")
    bound_provenance = {
        **provenance,
        "sample_count": len(file_names),
        "file_names_sha256": _names_sha256(file_names),
        "model_state_sha256": _state_sha256(model),
        "model_trainable_state_sha256": _state_sha256(model, trainable_only=True),
        "probe_state_sha256": _state_sha256(probes),
    }
    (root / "pcvl_provenance.json").write_text(json.dumps(bound_provenance, indent=2), encoding="utf-8")
    probes.train()
    return {**metrics, **decomposition}


def validate_pcvl_artifacts(path: str | Path, expected_identity: dict[str, Any] | None = None) -> None:
    root = Path(path)
    missing = [name for name in PCVL_FILES if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"Missing PCVL pilot artifacts: {missing}")
    json_files = [name for name in PCVL_FILES if name.endswith(".json")]
    loaded = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in json_files}
    if not _all_finite(loaded):
        raise RuntimeError("PCVL artifacts contain non-finite values")
    metrics = loaded["pcvl_metrics.json"]
    for key in ("u0_action_map", "u1_action_map", "u2_action_map", "u3_action_map", "predicate_action_value_supported"):
        if key not in metrics:
            raise RuntimeError(f"PCVL metrics missing {key}")
    per_action = loaded["pcvl_per_action.json"]
    if any(key not in per_action or len(per_action[key]) != 4 for key in ("u0", "u1", "u2", "u3")):
        raise RuntimeError("PCVL per-action schema is invalid")
    probabilities = torch.load(root / "pcvl_probabilities.pt", map_location="cpu", weights_only=True)
    labels = torch.load(root / "pcvl_labels.pt", map_location="cpu", weights_only=True)
    names = loaded["pcvl_file_names.json"].get("file_names", [])
    if set(probabilities) != {"u0", "u1", "u2", "u3"} or labels.ndim != 2 or labels.shape[1] != 4:
        raise RuntimeError("PCVL raw prediction schema is invalid")
    if any(value.shape != labels.shape or not torch.isfinite(value).all() for value in probabilities.values()) or not torch.isfinite(labels).all() or len(names) != len(labels):
        raise RuntimeError("PCVL raw predictions are incomplete or non-finite")
    for key, value in probabilities.items():
        recomputed = _average_precision(value, labels)
        recomputed_map = float(torch.nanmean(recomputed))
        if abs(recomputed_map - float(metrics[f"{key}_action_map"])) > 1e-8:
            raise RuntimeError(f"PCVL recomputed {key} mAP does not match aggregate metrics")
        for actual, recorded in zip(recomputed, per_action[key]):
            if torch.isnan(actual):
                if recorded is not None:
                    raise RuntimeError(f"PCVL recomputed {key} per-action AP mismatch")
            elif recorded is None or abs(float(actual) - float(recorded)) > 1e-8:
                raise RuntimeError(f"PCVL recomputed {key} per-action AP mismatch")
    bootstrap = loaded["pcvl_bootstrap.json"]
    for key in ("delta_value", "delta_measurement", "delta_interaction", "delta_learned_value", "delta_learned_interaction"):
        if key not in bootstrap or any(field not in bootstrap[key] for field in ("mean", "ci_low", "ci_high", "positive_rate")):
            raise RuntimeError(f"PCVL bootstrap schema missing {key}")
    generator = torch.Generator().manual_seed(20260722)
    raw_rows = []
    for _ in range(200):
        sample = torch.randint(0, len(labels), (len(labels),), generator=generator)
        sampled = {key: float(torch.nanmean(_average_precision(value[sample], labels[sample]))) for key, value in probabilities.items()}
        raw_rows.append({
            "delta_value": sampled["u1"] - sampled["u0"],
            "delta_measurement": sampled["u2"] - sampled["u1"],
            "delta_interaction": sampled["u3"] - sampled["u2"],
            "delta_learned_value": sampled["u2"] - sampled["u0"],
            "delta_learned_interaction": sampled["u3"] - sampled["u2"],
        })
    for key in raw_rows[0]:
        values = torch.tensor([row[key] for row in raw_rows])
        expected = {"mean": float(values.mean()), "ci_low": float(values.quantile(0.025)), "ci_high": float(values.quantile(0.975)), "positive_rate": float((values > 0).float().mean())}
        if any(abs(expected[field] - float(bootstrap[key][field])) > 1e-8 for field in expected):
            raise RuntimeError(f"PCVL recomputed bootstrap {key} does not match aggregate artifact")
    provenance = loaded["pcvl_provenance.json"]
    required_provenance = (
        "git_head", "source_tree_sha256", "config_sha256", "skill_sha256",
        "pretrained_weights_sha256", "action_schema_sha256",
        "train_audit_indices_sha256", "train_audit_file_names_sha256",
        "sample_count", "file_names_sha256", "model_state_sha256", "model_trainable_state_sha256",
        "probe_state_sha256", "epoch",
    )
    missing_provenance = [key for key in required_provenance if key not in provenance]
    if missing_provenance:
        raise RuntimeError(f"PCVL provenance missing {missing_provenance}")
    if provenance["file_names_sha256"] != provenance["train_audit_file_names_sha256"]:
        raise RuntimeError("PCVL provenance audit file order does not match the bound split")
    if provenance["sample_count"] != len(names) or provenance["file_names_sha256"] != _names_sha256(names):
        raise RuntimeError("PCVL raw sample identity does not match provenance")
    if expected_identity is not None:
        mismatched = [key for key, value in expected_identity.items() if provenance.get(key) != value]
        if mismatched:
            raise RuntimeError(f"PCVL provenance identity mismatch: {mismatched}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate_dir", required=True)
    args = parser.parse_args()
    validate_pcvl_artifacts(args.validate_dir)


if __name__ == "__main__":
    main()
