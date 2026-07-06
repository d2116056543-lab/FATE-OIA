from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from fate_oia.acpr_interactflow.psi_metrics import compute_psi_action_metrics, compute_psi_exp29_metrics
from fate_oia.acpr_interactflow.psi_sanity_dataset import PSISanityDataset, psi_sanity_collate
from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight) * mask
    if weights is not None:
        loss = loss * weights.view(-1, 1)
        denom = (mask * weights.view(-1, 1)).sum().clamp_min(1.0)
    else:
        denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def _action_kl(logits: torch.Tensor, soft: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    per = (soft.clamp_min(1e-9) * (soft.clamp_min(1e-9).log() - F.log_softmax(logits, dim=-1))).sum(dim=-1)
    if weights is not None:
        return (per * weights).sum() / weights.sum().clamp_min(1.0)
    return per.mean()


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def set_training_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_loader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(int(worker_seed))
    torch.manual_seed(int(worker_seed))


def action_rate_prior_loss(
    logits: torch.Tensor,
    target_prior: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    if weights is not None:
        pred_rate = (probs * weights.view(-1, 1)).sum(dim=0) / weights.sum().clamp_min(1.0)
    else:
        pred_rate = probs.mean(dim=0)
    target = target_prior.to(device=logits.device, dtype=logits.dtype)
    loss = (pred_rate - target).pow(2).sum()
    return loss, pred_rate


def _action_class_weights(dataset: PSISanityDataset) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.float32)
    for record, row in zip(dataset.records, dataset.protocol_index_rows):
        counts[int(dataset._record_action_id(record, row))] += 1.0
    weights = counts.sum().clamp_min(1.0) / (counts.clamp_min(1.0) * float(counts.numel()))
    return weights / weights.mean().clamp_min(1e-6)


def _action_class_rates(dataset: PSISanityDataset) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.float32)
    for record, row in zip(dataset.records, dataset.protocol_index_rows):
        counts[int(dataset._record_action_id(record, row))] += 1.0
    return counts / counts.sum().clamp_min(1.0)


def _parse_action_rate_prior(
    *,
    source: str,
    manual: str | None,
    train_dataset: PSISanityDataset,
) -> torch.Tensor:
    normalized = str(source or "none").strip().lower()
    if normalized in {"", "none", "off", "disabled"}:
        return torch.zeros(3, dtype=torch.float32)
    if normalized == "train":
        return _action_class_rates(train_dataset)
    if normalized == "manual":
        if not manual:
            raise ValueError("--action_rate_prior_manual is required when --action_rate_prior_source manual")
        values = [float(x.strip()) for x in manual.split(",") if x.strip()]
        if len(values) != 3:
            raise ValueError("--action_rate_prior_manual must contain exactly three comma-separated values")
        prior = torch.tensor(values, dtype=torch.float32)
        if not torch.isfinite(prior).all() or float(prior.sum().item()) <= 0.0:
            raise ValueError(f"Invalid action rate prior: {manual}")
        return prior / prior.sum().clamp_min(1e-6)
    raise ValueError(f"Unsupported --action_rate_prior_source: {source}")


def _exp29_training_stats(dataset: PSISanityDataset) -> dict[str, torch.Tensor | float]:
    positives = torch.zeros(29, dtype=torch.float32)
    observed = torch.zeros(29, dtype=torch.float32)
    cardinalities: list[float] = []
    for idx, record in enumerate(dataset.records):
        target, mask = dataset._exp29(idx, record)
        positives += target.float() * mask.float()
        observed += mask.float()
        if float(mask.sum().item()) > 0.0:
            cardinalities.append(float((target.float() * mask.float()).sum().item()))
    pos_rate = positives / observed.clamp_min(1.0)
    neg = (observed - positives).clamp_min(1.0)
    pos_weight = (neg / positives.clamp_min(1.0)).clamp(1.0, 12.0)
    return {
        "pos_rate": pos_rate.clamp(1e-4, 0.95),
        "pos_weight": pos_weight,
        "mean_cardinality": float(sum(cardinalities) / max(len(cardinalities), 1)),
        "observed_total": float(observed.sum().item()),
        "positive_total": float(positives.sum().item()),
    }


def _calibrate_global_shift_from_train_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    target_positive_rate: float,
) -> float:
    valid = mask > 0
    if int(valid.sum().item()) == 0:
        return 0.0
    values = logits[valid].float()
    target = float(max(1e-4, min(0.95, target_positive_rate)))
    lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = (lo + hi) * 0.5
        rate = float((torch.sigmoid(values + mid) >= 0.5).float().mean().item())
        if rate < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) * 0.5)


def resolve_exp_deploy_shift(
    *,
    mode: str,
    fixed_shift: float | None,
    auto_shift: float,
    resume_metrics: dict[str, Any] | None = None,
) -> float:
    """Resolve the deploy-time Exp29 logit shift for this epoch.

    `auto` recalibrates from the current epoch's train logits. `fixed` and
    `best_locked` are safer continuation modes after a good checkpoint because
    they prevent train-set positive-rate calibration from drifting every epoch.
    """
    normalized = str(mode).strip().lower()
    if normalized == "auto":
        return float(auto_shift)
    if normalized == "fixed":
        if fixed_shift is None:
            raise ValueError("--fixed_exp_deploy_shift is required when --exp_deploy_shift_mode fixed")
        return float(fixed_shift)
    if normalized == "best_locked":
        if fixed_shift is not None:
            return float(fixed_shift)
        metrics = resume_metrics or {}
        for key in ("exp29_train_deploy_shift", "exp29_deploy_shift"):
            if key in metrics and metrics[key] is not None:
                return float(metrics[key])
        return float(auto_shift)
    raise ValueError(f"Unsupported exp deploy shift mode: {mode}")


def get_primary_metric_value(metrics: dict[str, Any], primary_metric: str) -> float:
    if primary_metric not in metrics:
        raise KeyError(f"Primary metric {primary_metric!r} not found in metrics")
    value = metrics[primary_metric]
    if value is None:
        raise ValueError(f"Primary metric {primary_metric!r} is None")
    return float(value)


def select_metrics_for_best_checkpoint(
    *,
    best_split: str,
    test_metrics: dict[str, Any],
    val_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = str(best_split or "test").strip().lower()
    if normalized == "test":
        return test_metrics
    if normalized == "val":
        if val_metrics is None:
            raise ValueError("--best_split val requires --eval_val_split")
        return val_metrics
    raise ValueError(f"Unsupported best_split: {best_split!r}")


def should_stop_for_metric_regression(
    *,
    max_regression: float,
    current_joint: float | None = None,
    best_joint: float | None = None,
    current_metrics: dict[str, Any] | None = None,
    best_metrics: dict[str, Any] | None = None,
    primary_metric: str = "joint",
) -> bool:
    """Return True when the current epoch has regressed too far from best."""
    if current_metrics is not None or best_metrics is not None:
        if current_metrics is None or best_metrics is None:
            raise ValueError("current_metrics and best_metrics must be provided together")
        current_value = get_primary_metric_value(current_metrics, primary_metric)
        best_value = get_primary_metric_value(best_metrics, primary_metric)
    else:
        if current_joint is None or best_joint is None:
            raise ValueError("current_joint and best_joint are required when metrics are not provided")
        current_value = float(current_joint)
        best_value = float(best_joint)
    if best_value < 0:
        return False
    return current_value < best_value - float(max_regression)


def should_restore_resume_optimizer(
    *,
    checkpoint_has_optimizer: bool,
    reset_optimizer_on_resume: bool,
) -> bool:
    return bool(checkpoint_has_optimizer) and not bool(reset_optimizer_on_resume)


class PSIDinoProbeModel(nn.Module):
    """Frozen-DINO probe for PSI protocol learnability.

    This is intentionally not the formal CALI model. It answers one question:
    are frozen DINO visual features separable enough for the current PSI action
    and Exp29 labels when the protocol/masks are held fixed?
    """

    def __init__(
        self,
        input_mode: str,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        use_mock_dino: bool = False,
        mock_dim: int = 384,
        dino_input_size: tuple[int, int] = (320, 576),
        dino_chunk_size: int = 6,
        patch_size: int = 8,
        hidden_dim: int = 256,
        exp_bias_init: float = 0.0,
        temporal_pooler: str = "mean_delta",
        spatial_pooler: str = "mean",
        spatial_queries: int = 4,
    ) -> None:
        super().__init__()
        if input_mode not in PSISanityDataset.VALID_MODES:
            raise ValueError(f"Unsupported PSI input_mode: {input_mode}")
        if temporal_pooler not in {"mean_delta", "attention"}:
            raise ValueError(f"Unsupported temporal_pooler: {temporal_pooler}")
        if spatial_pooler not in {"mean", "attention"}:
            raise ValueError(f"Unsupported spatial_pooler: {spatial_pooler}")
        self.input_mode = input_mode
        self.temporal_pooler = temporal_pooler
        self.spatial_pooler = spatial_pooler
        self.spatial_queries_count = max(1, int(spatial_queries))
        self.dino_input_size = (int(dino_input_size[0]), int(dino_input_size[1]))
        self.dino_chunk_size = max(1, int(dino_chunk_size))
        self.dino = ACPRDinoFieldExtractor(
            patch_size=int(patch_size),
            selected_layers=tuple(int(x) for x in selected_layers),
            pretrained_weights=pretrained_weights,
            freeze_backbone=True,
            use_mock_dino=bool(use_mock_dino),
            mock_dim=int(mock_dim),
        )
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        self.dino.eval()
        if self.spatial_pooler == "attention":
            self.spatial_norm = nn.LayerNorm(self.dino.dim)
            self.spatial_queries = nn.Parameter(torch.randn(self.spatial_queries_count, self.dino.dim) * 0.02)
            self.feature_dim = self.dino.dim * (1 + self.spatial_queries_count)
        else:
            self.spatial_norm = None
            self.register_parameter("spatial_queries", None)
            self.feature_dim = self.dino.dim * 2
        pooled_dim = self.feature_dim * (4 if self.temporal_pooler == "attention" else 3)
        self.temporal_score = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.action_head = nn.Linear(hidden_dim, 3)
        self.exp29_head = nn.Linear(hidden_dim, 29)
        self.exp29_calibration_bias = nn.Parameter(torch.full((29,), float(exp_bias_init)))

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != self.dino_input_size:
            images = F.interpolate(images, size=self.dino_input_size, mode="bilinear", align_corners=False)
        feats: list[torch.Tensor] = []
        for chunk in images.split(self.dino_chunk_size, dim=0):
            with torch.no_grad():
                field = self.dino(chunk)
            cls_feat = field["cls_tokens_by_layer"].detach().mean(dim=1)
            patch_tokens_by_layer = field["patch_tokens_by_layer"].detach()
            if self.spatial_pooler == "attention":
                patch_tokens = patch_tokens_by_layer.reshape(patch_tokens_by_layer.shape[0], -1, patch_tokens_by_layer.shape[-1])
                norm_tokens = self.spatial_norm(patch_tokens)  # type: ignore[operator]
                scores = torch.einsum("qd,bnd->bqn", self.spatial_queries, norm_tokens) / math.sqrt(float(self.dino.dim))  # type: ignore[arg-type]
                attention = torch.softmax(scores, dim=-1)
                attended = torch.einsum("bqn,bnd->bqd", attention, patch_tokens).reshape(patch_tokens.shape[0], -1)
                feats.append(torch.cat([cls_feat, attended], dim=-1))
            else:
                patch_feat = patch_tokens_by_layer.mean(dim=(1, 2))
                feats.append(torch.cat([cls_feat, patch_feat], dim=-1))
        return torch.cat(feats, dim=0)

    def _pool_sequence(self, frame_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean_feat = frame_features.mean(dim=1)
        last_feat = frame_features[:, -1]
        delta_feat = frame_features[:, -1] - frame_features[:, 0]
        if self.temporal_pooler == "attention":
            scores = self.temporal_score(frame_features).squeeze(-1)
            attention = torch.softmax(scores, dim=1)
            attended = (frame_features * attention.unsqueeze(-1)).sum(dim=1)
            return torch.cat([mean_feat, last_feat, delta_feat, attended], dim=-1), attention
        return torch.cat([mean_feat, last_feat, delta_feat], dim=-1), None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal_attention: torch.Tensor | None = None
        if images.dim() == 5:
            bsz, frames, channels, height, width = images.shape
            flat = images.reshape(bsz * frames, channels, height, width)
            frame_features = self._encode_images(flat).reshape(bsz, frames, -1)
            probe_features, temporal_attention = self._pool_sequence(frame_features)
        elif images.dim() == 4:
            single = self._encode_images(images)
            zeros = torch.zeros_like(single)
            if self.temporal_pooler == "attention":
                probe_features = torch.cat([single, single, zeros, single], dim=-1)
            else:
                probe_features = torch.cat([single, single, zeros], dim=-1)
        else:
            raise ValueError(f"Expected [B,3,H,W] or [B,T,3,H,W], got {tuple(images.shape)}")
        hidden = self.head(probe_features)
        exp29_raw = self.exp29_head(hidden)
        exp29_calibrated = exp29_raw + self.exp29_calibration_bias.view(1, -1)
        out = {
            "action_logits": self.action_head(hidden),
            "exp29_logits_raw": exp29_raw,
            "exp29_logits": exp29_calibrated,
            "exp29_calibration_bias": self.exp29_calibration_bias,
            "probe_features": probe_features,
        }
        if temporal_attention is not None:
            out["temporal_attention"] = temporal_attention
        return out


def build_probe_manifest(
    *,
    input_mode: str,
    package_root: str,
    frames_root: str,
    protocol_index_dir: str | None,
    protocol_name: str | None,
    exp_supervision_policy: str,
    eval_exp_supervision_policy: str,
    dino_weights: str,
    selected_layers: tuple[int, ...],
    dino_input_size: tuple[int, int],
    temporal_pooler: str,
    spatial_pooler: str,
    spatial_queries: int,
    train_count: int,
    test_count: int,
    batch_size: int,
    num_workers: int,
    device: str,
    use_mock_dino: bool,
    use_decision_group_weight: bool,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "purpose": "frozen_dino_psi_learnability_probe",
        "input_mode": input_mode,
        "package_root": str(package_root),
        "frames_root": str(frames_root),
        "protocol_index_dir": str(protocol_index_dir) if protocol_index_dir else None,
        "protocol_name": protocol_name,
        "exp_supervision_policy": exp_supervision_policy,
        "eval_exp_supervision_policy": eval_exp_supervision_policy,
        "dino_weights": str(dino_weights),
        "selected_layers": [int(x) for x in selected_layers],
        "dino_input_size": [int(dino_input_size[0]), int(dino_input_size[1])],
        "temporal_pooler": str(temporal_pooler),
        "spatial_pooler": str(spatial_pooler),
        "spatial_queries": int(spatial_queries),
        "train_count": int(train_count),
        "test_count": int(test_count),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "device": str(device),
        "use_mock_dino": bool(use_mock_dino),
        "use_decision_group_weight": bool(use_decision_group_weight),
        "seed": int(seed) if seed is not None else None,
        "loader_seed": int(seed) if seed is not None else None,
        "dino_frozen": True,
        "feature_cache_enabled": False,
        "token_cache_enabled": False,
        "logit_cache_enabled": False,
        "formal_result": False,
    }


def _loader(dataset: PSISanityDataset, batch_size: int, workers: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": workers > 0,
        "collate_fn": psi_sanity_collate,
    }
    generator = make_loader_generator(seed)
    if generator is not None:
        kwargs["generator"] = generator
        kwargs["worker_init_fn"] = seed_loader_worker
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate(
    model: PSIDinoProbeModel,
    loader: DataLoader,
    device: torch.device,
    exp_logit_shift: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    action_logits: list[torch.Tensor] = []
    exp_logits: list[torch.Tensor] = []
    exp_logits_raw: list[torch.Tensor] = []
    action_soft: list[torch.Tensor] = []
    action_majority: list[torch.Tensor] = []
    exp_targets: list[torch.Tensor] = []
    exp_masks: list[torch.Tensor] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        out = model(images)
        action_logits.append(out["action_logits"].detach().cpu())
        exp_logits.append(out["exp29_logits"].detach().cpu())
        exp_logits_raw.append(out["exp29_logits_raw"].detach().cpu())
        action_soft.append(batch["action_soft"].detach().cpu())
        action_majority.append(batch["action_majority"].detach().cpu())
        exp_targets.append(batch["exp29"].detach().cpu())
        exp_masks.append(batch["exp29_mask"].detach().cpu())
    z_action = torch.cat(action_logits, dim=0)
    z_exp_calibrated = torch.cat(exp_logits, dim=0)
    z_exp = z_exp_calibrated + float(exp_logit_shift)
    z_exp_raw = torch.cat(exp_logits_raw, dim=0)
    y_soft = torch.cat(action_soft, dim=0)
    y_major = torch.cat(action_majority, dim=0)
    y_exp = torch.cat(exp_targets, dim=0)
    m_exp = torch.cat(exp_masks, dim=0)
    action = compute_psi_action_metrics(z_action, y_major, y_soft)
    exp = compute_psi_exp29_metrics(z_exp, y_exp, m_exp)
    exp_raw = compute_psi_exp29_metrics(z_exp_raw, y_exp, m_exp)
    return {
        **action,
        **exp,
        "action": action,
        "exp29": exp,
        "exp29_raw": exp_raw,
        "joint": 0.70 * float(action["Act_mAcc"]) + 0.30 * float(exp["Exp_mF1"]),
        "exp29_pred_positive_rate_0p5": float((torch.sigmoid(z_exp) >= 0.5).float().mean().item()),
        "exp29_prob_mean": float(torch.sigmoid(z_exp).mean().item()),
        "exp29_deploy_shift": float(exp_logit_shift),
        "ExpRaw_mF1": exp_raw.get("Exp_mF1"),
        "ExpRaw_oF1": exp_raw.get("Exp_oF1"),
        "ExpRaw_mAP": exp_raw.get("Exp_mAP"),
        "ExpCal_mF1": exp.get("Exp_mF1"),
        "ExpCal_oF1": exp.get("Exp_oF1"),
        "ExpCal_mAP": exp.get("Exp_mAP"),
        "exp29_raw_pred_positive_rate_0p5": float((torch.sigmoid(z_exp_raw) >= 0.5).float().mean().item()),
        "exp29_raw_prob_mean": float(torch.sigmoid(z_exp_raw).mean().item()),
        "exp29_calibrated_pred_positive_rate_0p5": float((torch.sigmoid(z_exp_calibrated) >= 0.5).float().mean().item()),
        "exp29_calibrated_prob_mean": float(torch.sigmoid(z_exp_calibrated).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-DINO PSI learnability probe; no cache, not formal training.")
    parser.add_argument("--config")
    parser.add_argument("--package_root")
    parser.add_argument("--frames_root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_mode", choices=sorted(PSISanityDataset.VALID_MODES), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--temporal_pooler", choices=("mean_delta", "attention"), default="mean_delta")
    parser.add_argument("--spatial_pooler", choices=("mean", "attention"), default="mean")
    parser.add_argument("--spatial_queries", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_sample_strategy")
    parser.add_argument("--eval_max_sample_strategy")
    parser.add_argument("--max_sample_seed", type=int)
    parser.add_argument("--protocol_name")
    parser.add_argument("--protocol_index_dir")
    parser.add_argument(
        "--disable_protocol_index",
        action="store_true",
        help="Ignore any protocol_index configured in YAML and use the full package split.",
    )
    parser.add_argument("--exp_supervision_policy")
    parser.add_argument("--eval_exp_supervision_policy")
    parser.add_argument("--exp_near_keyframe_max_gap", type=int)
    parser.add_argument(
        "--use_decision_group_weight",
        action="store_true",
        help="Train with 1/N event weights for expanded rows from the same video_id::decision_keyframe.",
    )
    parser.add_argument("--dino_weights")
    parser.add_argument("--dino_input_height", type=int)
    parser.add_argument("--dino_input_width", type=int)
    parser.add_argument("--dino_chunk_size", type=int)
    parser.add_argument("--selected_layers", default="")
    parser.add_argument("--use_mock_dino", action="store_true")
    parser.add_argument("--mock_dim", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, help="Set Python/Torch seeds for reproducible probe initialization and shuffling.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest_probe.pth in output_dir unless --resume_path is set.")
    parser.add_argument("--resume_path", help="Optional explicit checkpoint path for --resume.")
    parser.add_argument("--reset_optimizer_on_resume", action="store_true", help="Load model weights but start a fresh optimizer with the requested LR.")
    parser.add_argument("--log_interval", type=int, default=50, help="Write batch-level loss/progress every N optimizer steps.")
    parser.add_argument("--action_ce_weight", type=float, default=0.35)
    parser.add_argument("--action_rate_prior_weight", type=float, default=0.0)
    parser.add_argument(
        "--action_rate_prior_source",
        default="none",
        help="Action prediction-rate prior source: none, train, or manual.",
    )
    parser.add_argument(
        "--action_rate_prior_manual",
        help="Comma-separated Maintain,Reduce,Stop prior used when --action_rate_prior_source manual.",
    )
    parser.add_argument("--exp_loss_weight", type=float, default=0.35)
    parser.add_argument("--exp_positive_rate_weight", type=float, default=0.08)
    parser.add_argument("--exp_cardinality_weight", type=float, default=0.04)
    parser.add_argument("--exp_bias_init", type=float, default=1.15)
    parser.add_argument("--auto_exp_deploy_shift", action="store_true")
    parser.add_argument(
        "--exp_deploy_shift_mode",
        choices=("auto", "fixed", "best_locked"),
        default="auto",
        help="Resolve deploy shift from current train logits, fixed value, or the resumed best checkpoint.",
    )
    parser.add_argument("--fixed_exp_deploy_shift", type=float)
    parser.add_argument("--stop_on_regression", action="store_true")
    parser.add_argument("--max_joint_regression", type=float, default=0.003)
    parser.add_argument(
        "--primary_metric",
        default="joint",
        help="Metric used for best checkpoint selection and regression guard, e.g. joint or Act_mAcc.",
    )
    parser.add_argument(
        "--eval_val_split",
        action="store_true",
        help="Evaluate val after each epoch. Pair with --best_split val for event-dev early stopping.",
    )
    parser.add_argument(
        "--best_split",
        choices=("test", "val"),
        default="test",
        help="Split used for best checkpoint selection and regression guard.",
    )
    parser.add_argument("--freeze_action_path", action="store_true", help="Freeze shared/action path and train only Exp29 head/calibration.")
    parser.add_argument(
        "--eval_train_split",
        action="store_true",
        help="Diagnostic only: evaluate the training split after each epoch to distinguish overfit capacity from cross-video generalization.",
    )
    args = parser.parse_args()
    set_training_seed(args.seed)

    cfg = _load_yaml(args.config)
    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    visual_cfg = cfg.get("model", {}).get("visual_encoder", {})
    package_root = args.package_root or paths.get("psi_package_root")
    frames_root = args.frames_root or paths.get("psi2_root_reference_only") or paths.get("psi1_root")
    if not package_root or not frames_root:
        raise ValueError("package_root and frames_root are required, either by args or config")
    protocol_cfg = data.get("protocol_index", {}) if isinstance(data.get("protocol_index", {}), dict) else {}
    protocol_enabled = bool(protocol_cfg.get("enabled", False)) and not bool(args.disable_protocol_index)
    protocol_index_dir = None if args.disable_protocol_index else (args.protocol_index_dir or (protocol_cfg.get("dir") if protocol_enabled else None))
    protocol_name = None if args.disable_protocol_index else (args.protocol_name or (protocol_cfg.get("name") if protocol_enabled else None))
    exp_policy = args.exp_supervision_policy or data.get("exp_supervision_policy", "record_mask")
    eval_exp_policy = args.eval_exp_supervision_policy or data.get("eval_exp_supervision_policy", exp_policy)
    exp_gap = int(args.exp_near_keyframe_max_gap or data.get("exp_near_keyframe_max_gap", 30))
    train_sample_strategy = args.max_sample_strategy or data.get("max_sample_strategy", "head")
    eval_sample_strategy = args.eval_max_sample_strategy or data.get("eval_max_sample_strategy", train_sample_strategy)
    max_sample_seed = int(args.max_sample_seed or data.get("max_sample_seed", 7))
    selected_layers = tuple(int(x) for x in (args.selected_layers.split(",") if args.selected_layers else visual_cfg.get("selected_layers", [3, 7, 11])))
    dino_input_size = (
        int(args.dino_input_height or visual_cfg.get("dino_input_height", data.get("image_height", 320))),
        int(args.dino_input_width or visual_cfg.get("dino_input_width", data.get("image_width", 576))),
    )
    dino_chunk_size = int(args.dino_chunk_size or visual_cfg.get("dino_chunk_size", 6))
    dino_weights = args.dino_weights or paths.get("dino_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_dataset_kwargs = {
        "frames_root": frames_root,
        "image_size": dino_input_size,
        "protocol_index_dir": protocol_index_dir,
        "protocol_name": protocol_name,
        "exp_near_keyframe_max_gap": exp_gap,
        "max_sample_seed": max_sample_seed,
    }
    train_ds = PSISanityDataset(
        package_root,
        "train",
        args.input_mode,
        max_samples=args.max_train_samples,
        max_sample_strategy=train_sample_strategy,
        exp_supervision_policy=exp_policy,
        use_decision_group_weight=args.use_decision_group_weight,
        **common_dataset_kwargs,
    )
    test_ds = PSISanityDataset(
        package_root,
        "test",
        args.input_mode,
        max_samples=args.max_test_samples,
        max_sample_strategy=eval_sample_strategy,
        exp_supervision_policy=eval_exp_policy,
        **common_dataset_kwargs,
    )
    val_ds = None
    if args.eval_val_split:
        val_ds = PSISanityDataset(
            package_root,
            "val",
            args.input_mode,
            max_samples=args.max_test_samples,
            max_sample_strategy=eval_sample_strategy,
            exp_supervision_policy=eval_exp_policy,
            **common_dataset_kwargs,
        )
    if args.best_split == "val" and val_ds is None:
        raise ValueError("--best_split val requires --eval_val_split")
    train_loader = _loader(train_ds, args.batch_size, args.num_workers, shuffle=True, seed=args.seed)
    test_loader = _loader(test_ds, args.batch_size, args.num_workers, shuffle=False, seed=args.seed)
    val_loader = _loader(val_ds, args.batch_size, args.num_workers, shuffle=False, seed=args.seed) if val_ds is not None else None
    action_class_weights = _action_class_weights(train_ds).to(device)
    action_rate_prior = _parse_action_rate_prior(
        source=args.action_rate_prior_source,
        manual=args.action_rate_prior_manual,
        train_dataset=train_ds,
    ).to(device)
    exp_stats = _exp29_training_stats(train_ds)
    exp_pos_weight = exp_stats["pos_weight"].to(device)  # type: ignore[union-attr]
    target_exp_positive_rate = float(exp_stats["positive_total"]) / max(float(exp_stats["observed_total"]), 1.0)
    target_exp_cardinality = float(exp_stats["mean_cardinality"])
    model = PSIDinoProbeModel(
        input_mode=args.input_mode,
        selected_layers=selected_layers,
        pretrained_weights=dino_weights,
        use_mock_dino=args.use_mock_dino,
        mock_dim=args.mock_dim,
        dino_input_size=dino_input_size,
        dino_chunk_size=dino_chunk_size,
        hidden_dim=args.hidden_dim,
        exp_bias_init=args.exp_bias_init,
        temporal_pooler=args.temporal_pooler,
        spatial_pooler=args.spatial_pooler,
        spatial_queries=args.spatial_queries,
    ).to(device)
    if args.freeze_action_path:
        for parameter in model.head.parameters():
            parameter.requires_grad = False
        for parameter in model.action_head.parameters():
            parameter.requires_grad = False
        for parameter in model.exp29_head.parameters():
            parameter.requires_grad = True
        model.exp29_calibration_bias.requires_grad = True
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters selected for PSI DINO probe")
    opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.01)
    start_epoch = 0
    best_joint = -1.0
    best_metric_value = -1.0
    best_metrics: dict[str, Any] = {}
    resume_payload: dict[str, Any] = {}
    resume_metrics: dict[str, Any] = {}
    if args.resume or args.resume_path:
        resume_path = Path(args.resume_path) if args.resume_path else out_dir / "checkpoint_latest_probe.pth"
        if not resume_path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device)
        missing, unexpected = model.load_state_dict(resume_payload["model"], strict=False)
        if should_restore_resume_optimizer(
            checkpoint_has_optimizer="optimizer" in resume_payload,
            reset_optimizer_on_resume=bool(args.reset_optimizer_on_resume),
        ):
            try:
                opt.load_state_dict(resume_payload["optimizer"])
                optimizer_restored = True
            except ValueError:
                optimizer_restored = False
        else:
            optimizer_restored = False
        resume_epoch = int(resume_payload.get("epoch", -1))
        start_epoch = resume_epoch + 1
        resume_metrics = resume_payload.get("metrics") or {}
        best_joint = float(resume_payload.get("best_joint", resume_metrics.get("joint", -1.0)))
        best_metrics = resume_payload.get("best_metrics") or resume_metrics
        best_metric_value = float(
            resume_payload.get(
                "best_metric_value",
                best_metrics.get(args.primary_metric, best_joint),
            )
        )
        print(
            "dino_probe_resume "
            f"path={resume_path} start_epoch={start_epoch} best_joint={best_joint:.4f} "
            f"best_{args.primary_metric}={best_metric_value:.4f} "
            f"optimizer_restored={optimizer_restored} missing={list(missing)} unexpected={list(unexpected)}",
            flush=True,
        )
    manifest = build_probe_manifest(
        input_mode=args.input_mode,
        package_root=str(package_root),
        frames_root=str(frames_root),
        protocol_index_dir=str(protocol_index_dir) if protocol_index_dir else None,
        protocol_name=protocol_name,
        exp_supervision_policy=exp_policy,
        eval_exp_supervision_policy=eval_exp_policy,
        dino_weights=str(dino_weights),
        selected_layers=selected_layers,
        dino_input_size=dino_input_size,
        temporal_pooler=args.temporal_pooler,
        spatial_pooler=args.spatial_pooler,
        spatial_queries=args.spatial_queries,
        train_count=len(train_ds),
        test_count=len(test_ds),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        use_mock_dino=args.use_mock_dino,
        use_decision_group_weight=args.use_decision_group_weight,
        seed=args.seed,
    )
    manifest["training_adjustments"] = {
        "action_ce_weight": float(args.action_ce_weight),
        "action_class_weights": [float(x) for x in action_class_weights.detach().cpu().tolist()],
        "action_rate_prior_weight": float(args.action_rate_prior_weight),
        "action_rate_prior_source": str(args.action_rate_prior_source),
        "action_rate_prior": [float(x) for x in action_rate_prior.detach().cpu().tolist()],
        "exp_loss_weight": float(args.exp_loss_weight),
        "exp_positive_rate_weight": float(args.exp_positive_rate_weight),
        "exp_cardinality_weight": float(args.exp_cardinality_weight),
        "exp_bias_init": float(args.exp_bias_init),
        "auto_exp_deploy_shift": bool(args.auto_exp_deploy_shift),
        "exp_deploy_shift_mode": str(args.exp_deploy_shift_mode),
        "fixed_exp_deploy_shift": None if args.fixed_exp_deploy_shift is None else float(args.fixed_exp_deploy_shift),
        "stop_on_regression": bool(args.stop_on_regression),
        "max_joint_regression": float(args.max_joint_regression),
        "primary_metric": str(args.primary_metric),
        "best_split": str(args.best_split),
        "eval_val_split": bool(args.eval_val_split),
        "val_count": int(len(val_ds)) if val_ds is not None else 0,
        "freeze_action_path": bool(args.freeze_action_path),
        "spatial_pooler": str(args.spatial_pooler),
        "spatial_queries": int(args.spatial_queries),
        "disable_protocol_index": bool(args.disable_protocol_index),
        "use_decision_group_weight": bool(args.use_decision_group_weight),
        "trainable_parameter_count": int(sum(p.numel() for p in trainable_parameters)),
        "exp_pos_weight_mean": float(exp_pos_weight.detach().cpu().mean().item()),
        "target_exp_positive_rate": target_exp_positive_rate,
        "target_exp_cardinality": target_exp_cardinality,
    }
    if resume_payload:
        manifest["resume"] = {
            "enabled": True,
            "path": str(Path(args.resume_path) if args.resume_path else out_dir / "checkpoint_latest_probe.pth"),
            "loaded_epoch": int(resume_payload.get("epoch", -1)),
            "start_epoch": int(start_epoch),
            "optimizer_restored": bool(optimizer_restored),
            "reset_optimizer_on_resume": bool(args.reset_optimizer_on_resume),
        }
    else:
        manifest["resume"] = {"enabled": False}
    _write_json(out_dir / "run_manifest.json", manifest)
    if start_epoch >= args.epochs:
        print(f"dino_probe_resume already_complete start_epoch={start_epoch} requested_epochs={args.epochs}", flush=True)
        return
    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_steps = len(train_loader)
        train_exp_logits_for_shift: list[torch.Tensor] = []
        train_exp_masks_for_shift: list[torch.Tensor] = []
        for step, batch in enumerate(train_loader, start=1):
            images = batch["images"].to(device, non_blocking=True)
            y_action = batch["action_soft"].to(device, non_blocking=True)
            y_major = batch["action_majority"].to(device, non_blocking=True)
            y_exp = batch["exp29"].to(device, non_blocking=True)
            m_exp = batch["exp29_mask"].to(device, non_blocking=True)
            weights = batch["paper_effective_weight"].to(device, non_blocking=True)
            out = model(images)
            if args.auto_exp_deploy_shift:
                train_exp_logits_for_shift.append(out["exp29_logits"].detach().cpu())
                train_exp_masks_for_shift.append(m_exp.detach().cpu())
            loss_action_kl = _action_kl(out["action_logits"], y_action, weights)
            action_ce_per = F.cross_entropy(out["action_logits"], y_major, weight=action_class_weights, reduction="none")
            loss_action_ce = _weighted_mean(action_ce_per, weights)
            if float(args.action_rate_prior_weight) > 0.0:
                loss_action_rate_prior, pred_action_rate = action_rate_prior_loss(out["action_logits"], action_rate_prior, weights)
            else:
                loss_action_rate_prior = loss_action_kl * 0.0
                _, pred_action_rate = action_rate_prior_loss(out["action_logits"], action_rate_prior, weights)
            loss_action = (
                loss_action_kl
                + float(args.action_ce_weight) * loss_action_ce
                + float(args.action_rate_prior_weight) * loss_action_rate_prior
            )
            loss_exp = _masked_bce(out["exp29_logits"], y_exp, m_exp, weights, exp_pos_weight)
            exp_probs = torch.sigmoid(out["exp29_logits"])
            valid_exp = (m_exp.sum(dim=1) > 0).float()
            pred_positive_rate = (exp_probs * m_exp).sum() / m_exp.sum().clamp_min(1.0)
            loss_exp_positive_rate = (pred_positive_rate - target_exp_positive_rate) ** 2
            pred_cardinality = (exp_probs * m_exp).sum(dim=1)
            target_cardinality = (y_exp * m_exp).sum(dim=1)
            if float(valid_exp.sum().detach().cpu().item()) > 0.0:
                loss_exp_cardinality = (((pred_cardinality - target_cardinality) ** 2) * valid_exp).sum() / valid_exp.sum().clamp_min(1.0)
            else:
                loss_exp_cardinality = pred_cardinality.sum() * 0.0
            if args.freeze_action_path:
                loss = (
                    float(args.exp_loss_weight) * loss_exp
                    + float(args.exp_positive_rate_weight) * loss_exp_positive_rate
                    + float(args.exp_cardinality_weight) * loss_exp_cardinality
                )
            else:
                loss = (
                    loss_action
                    + float(args.exp_loss_weight) * loss_exp
                    + float(args.exp_positive_rate_weight) * loss_exp_positive_rate
                    + float(args.exp_cardinality_weight) * loss_exp_cardinality
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            opt.step()
            total_loss += float(loss.detach().cpu().item())
            if args.log_interval > 0 and (step == 1 or step % args.log_interval == 0 or step == total_steps):
                gpu_peak_gib = 0.0
                if device.type == "cuda":
                    gpu_peak_gib = float(torch.cuda.max_memory_reserved(device) / (1024**3))
                row = {
                    "epoch": int(epoch),
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "loss_total": float(loss.detach().cpu().item()),
                    "loss_action": float(loss_action.detach().cpu().item()),
                    "loss_action_kl": float(loss_action_kl.detach().cpu().item()),
                    "loss_action_ce": float(loss_action_ce.detach().cpu().item()),
                    "loss_action_rate_prior": float(loss_action_rate_prior.detach().cpu().item()),
                    "pred_action_rate": [float(x) for x in pred_action_rate.detach().cpu().tolist()],
                    "target_action_rate_prior": [float(x) for x in action_rate_prior.detach().cpu().tolist()],
                    "loss_exp": float(loss_exp.detach().cpu().item()),
                    "loss_exp_positive_rate": float(loss_exp_positive_rate.detach().cpu().item()),
                    "loss_exp_cardinality": float(loss_exp_cardinality.detach().cpu().item()),
                    "pred_exp_positive_rate": float(pred_positive_rate.detach().cpu().item()),
                    "target_exp_positive_rate": float(target_exp_positive_rate),
                    "gpu_peak_reserved_gib": gpu_peak_gib,
                    "elapsed_seconds": float(time.time() - start),
                }
                _append_jsonl(out_dir / "loss_components.jsonl", row)
                print(
                    "dino_probe_batch "
                    f"epoch={epoch} step={step}/{total_steps} loss={row['loss_total']:.4f} "
                    f"action={row['loss_action']:.4f} exp={row['loss_exp']:.4f} "
                    f"gpuPeakGiB={gpu_peak_gib:.2f} elapsed={row['elapsed_seconds']:.1f}",
                    flush=True,
                )
        exp_deploy_shift = 0.0
        if args.auto_exp_deploy_shift and train_exp_logits_for_shift:
            exp_deploy_shift = _calibrate_global_shift_from_train_logits(
                torch.cat(train_exp_logits_for_shift, dim=0),
                torch.cat(train_exp_masks_for_shift, dim=0),
                target_exp_positive_rate,
            )
        auto_exp_deploy_shift = exp_deploy_shift
        exp_deploy_shift = resolve_exp_deploy_shift(
            mode=args.exp_deploy_shift_mode,
            fixed_shift=args.fixed_exp_deploy_shift,
            auto_shift=auto_exp_deploy_shift,
            resume_metrics=resume_metrics,
        )
        metrics = evaluate(model, test_loader, device, exp_logit_shift=exp_deploy_shift)
        metrics["split"] = "test"
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, exp_logit_shift=exp_deploy_shift)
            val_metrics["split"] = "val"
            val_metrics["epoch"] = epoch
            _append_jsonl(out_dir / "val_metrics_summary.jsonl", val_metrics)
            _append_jsonl(
                out_dir / "val_core_metrics_summary.jsonl",
                {
                    "epoch": epoch,
                    "Act_mAcc": val_metrics.get("Act_mAcc"),
                    "Act_oAcc": val_metrics.get("Act_oAcc"),
                    "Act_macroF1": val_metrics.get("Act_macroF1"),
                    "Maintain_F1": val_metrics.get("Maintain_F1"),
                    "Reduce_F1": val_metrics.get("Reduce_F1"),
                    "Stop_F1": val_metrics.get("Stop_F1"),
                    "Exp_mF1": val_metrics.get("Exp_mF1"),
                    "Exp_oF1": val_metrics.get("Exp_oF1"),
                    "Exp_mAP": val_metrics.get("Exp_mAP"),
                    "ExpCal_mF1": val_metrics.get("ExpCal_mF1"),
                    "ExpCal_oF1": val_metrics.get("ExpCal_oF1"),
                    "ExpCal_mAP": val_metrics.get("ExpCal_mAP"),
                    "joint": val_metrics.get("joint"),
                },
            )
        train_metrics = None
        if args.eval_train_split:
            train_eval_loader = _loader(train_ds, args.batch_size, args.num_workers, shuffle=False, seed=args.seed)
            train_metrics = evaluate(model, train_eval_loader, device, exp_logit_shift=exp_deploy_shift)
            train_metrics["epoch"] = epoch
            _append_jsonl(out_dir / "train_metrics_summary.jsonl", train_metrics)
        metrics["epoch"] = epoch
        metrics["train_loss_mean"] = total_loss / max(len(train_loader), 1)
        metrics["epoch_seconds"] = time.time() - start
        metrics["exp29_train_deploy_shift"] = exp_deploy_shift
        metrics["exp29_auto_train_deploy_shift"] = auto_exp_deploy_shift
        metrics["exp29_deploy_shift_mode"] = args.exp_deploy_shift_mode
        metrics["best_split"] = args.best_split
        if val_metrics is not None:
            metrics["Val_Act_mAcc"] = val_metrics.get("Act_mAcc")
            metrics["Val_Act_oAcc"] = val_metrics.get("Act_oAcc")
            metrics["Val_Act_macroF1"] = val_metrics.get("Act_macroF1")
            metrics["Val_ExpCal_mF1"] = val_metrics.get("ExpCal_mF1")
            metrics["Val_ExpCal_oF1"] = val_metrics.get("ExpCal_oF1")
            metrics["Val_joint"] = val_metrics.get("joint")
        if train_metrics is not None:
            metrics["Train_Act_mAcc"] = train_metrics.get("Act_mAcc")
            metrics["Train_Act_oAcc"] = train_metrics.get("Act_oAcc")
            metrics["Train_Act_macroF1"] = train_metrics.get("Act_macroF1")
            metrics["Train_ExpCal_mF1"] = train_metrics.get("ExpCal_mF1")
            metrics["Train_ExpCal_oF1"] = train_metrics.get("ExpCal_oF1")
        _append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
        _append_jsonl(
            out_dir / "core_metrics_summary.jsonl",
            {
                "epoch": epoch,
                "Act_mAcc": metrics.get("Act_mAcc"),
                "Act_oAcc": metrics.get("Act_oAcc"),
                "Act_macroF1": metrics.get("Act_macroF1"),
                "Maintain_F1": metrics.get("Maintain_F1"),
                "Reduce_F1": metrics.get("Reduce_F1"),
                "Stop_F1": metrics.get("Stop_F1"),
                "Exp_mF1": metrics.get("Exp_mF1"),
                "Exp_oF1": metrics.get("Exp_oF1"),
                "Exp_mAP": metrics.get("Exp_mAP"),
                "ExpRaw_mF1": metrics.get("ExpRaw_mF1"),
                "ExpRaw_oF1": metrics.get("ExpRaw_oF1"),
                "ExpRaw_mAP": metrics.get("ExpRaw_mAP"),
                "ExpCal_mF1": metrics.get("ExpCal_mF1"),
                "ExpCal_oF1": metrics.get("ExpCal_oF1"),
                "ExpCal_mAP": metrics.get("ExpCal_mAP"),
                "exp29_pred_positive_rate_0p5": metrics.get("exp29_pred_positive_rate_0p5"),
                "exp29_prob_mean": metrics.get("exp29_prob_mean"),
                "exp29_deploy_shift": metrics.get("exp29_deploy_shift"),
                "exp29_calibrated_pred_positive_rate_0p5": metrics.get("exp29_calibrated_pred_positive_rate_0p5"),
                "exp29_calibrated_prob_mean": metrics.get("exp29_calibrated_prob_mean"),
                "exp29_raw_pred_positive_rate_0p5": metrics.get("exp29_raw_pred_positive_rate_0p5"),
                "exp29_raw_prob_mean": metrics.get("exp29_raw_prob_mean"),
                "joint": metrics.get("joint"),
                "best_split": metrics.get("best_split"),
                "Val_Act_mAcc": metrics.get("Val_Act_mAcc"),
                "Val_Act_oAcc": metrics.get("Val_Act_oAcc"),
                "Val_Act_macroF1": metrics.get("Val_Act_macroF1"),
                "Val_ExpCal_mF1": metrics.get("Val_ExpCal_mF1"),
                "Val_joint": metrics.get("Val_joint"),
                "train_loss_mean": metrics.get("train_loss_mean"),
                "epoch_seconds": metrics.get("epoch_seconds"),
            },
        )
        selection_metrics = select_metrics_for_best_checkpoint(
            best_split=args.best_split,
            test_metrics=metrics,
            val_metrics=val_metrics,
        )
        current_metric_value = get_primary_metric_value(selection_metrics, args.primary_metric)
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_metrics = dict(selection_metrics)
            best_joint = float(metrics["joint"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "selection_metrics": selection_metrics,
                    "manifest": manifest,
                    "best_joint": best_joint,
                    "best_split": args.best_split,
                    "best_metric_name": args.primary_metric,
                    "best_metric_value": best_metric_value,
                    "best_metrics": best_metrics,
                },
                out_dir / "checkpoint_best_probe.pth",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "selection_metrics": selection_metrics,
                "manifest": manifest,
                "best_joint": best_joint,
                "best_split": args.best_split,
                "best_metric_name": args.primary_metric,
                "best_metric_value": best_metric_value,
                "best_metrics": best_metrics,
            },
            out_dir / "checkpoint_latest_probe.pth",
        )
        if args.stop_on_regression and should_stop_for_metric_regression(
            current_metrics=selection_metrics,
            best_metrics=best_metrics,
            primary_metric=args.primary_metric,
            max_regression=float(args.max_joint_regression),
        ):
            print(
                "dino_probe_stop_on_regression "
                f"epoch={epoch} {args.primary_metric}={current_metric_value:.4f} "
                f"best_{args.primary_metric}={best_metric_value:.4f} "
                f"split={args.best_split} joint={float(metrics['joint']):.4f} best_joint={best_joint:.4f} "
                f"max_regression={float(args.max_joint_regression):.4f}",
                flush=True,
            )
            break
        message = (
            "dino_probe "
            f"epoch={epoch} Act_mAcc={float(metrics['Act_mAcc']):.4f} Act_oAcc={float(metrics['Act_oAcc']):.4f} "
        )
        if val_metrics is not None:
            message += f"Val_Act_mAcc={float(val_metrics['Act_mAcc']):.4f} Val_Act_oAcc={float(val_metrics['Act_oAcc']):.4f} "
        message += (
            f"Exp_mF1={float(metrics['Exp_mF1']):.4f} Exp_oF1={float(metrics['Exp_oF1']):.4f} "
            f"Exp_mAP={float(metrics['Exp_mAP']):.4f} joint={float(metrics['joint']):.4f} sec={metrics['epoch_seconds']:.1f}"
        )
        print(message, flush=True)


if __name__ == "__main__":
    main()
