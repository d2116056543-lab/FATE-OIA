from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex, load_bdd100k_objects
from fate_oia.grounding.mask_builder import drivable_map_to_mask, objects_to_mask
from fate_oia.losses import acpr_losses as L
from fate_oia.models.acpr_oia_model import ACPROIAModel
from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder
from fate_oia.models.acpr_reason_grammar import ACPRReasonGrammar
from fate_oia.models.acpr_visual_token_adapter import ACPRPredicateAnchoredVisualAdapter, VistaScaleSchedule
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices
from fate_oia.engine.train_acpr_oia import (
    build_model,
    collect_threshold_teacher,
    load_config,
    make_dataset,
    make_loader,
    reason_predicate_matrices,
)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _gate_a(device: torch.device) -> dict:
    torch.manual_seed(11)
    adapter = ACPRPredicateAnchoredVisualAdapter(dim=384, rank=48, num_layers=3, num_predicates=32).to(device)
    x = torch.randn(1, 3, 3600, 384, device=device)
    probs = torch.rand(1, 32, device=device)
    attn = torch.softmax(torch.randn(1, 32, 3600, device=device), dim=-1)
    y, stats = adapter(x, probs, attn, epoch=0)
    max_abs = float((y - x).abs().max().detach().cpu())
    return {
        "pass": max_abs <= 1e-6,
        "max_abs_diff": max_abs,
        "vista_alpha_abs_mean": float(stats["vista_alpha_abs_mean"].detach().cpu()),
        "vista_anchor_mix": float(stats["vista_anchor_mix"]),
        "check": "zero_up_equivalence_without_bypass",
    }


def _gate_b(device: torch.device) -> dict:
    torch.manual_seed(13)
    adapter = ACPRPredicateAnchoredVisualAdapter(dim=384, rank=48, num_layers=3, num_predicates=32).to(device)
    opt = torch.optim.SGD(adapter.parameters(), lr=0.5)
    x = torch.randn(1, 3, 3600, 384, device=device)
    target = x + 0.05 * torch.tanh(x.roll(shifts=1, dims=2))
    probs = torch.rand(1, 32, device=device)
    attn = torch.softmax(torch.randn(1, 32, 3600, device=device), dim=-1)
    y, _ = adapter(x, probs, attn, epoch=0)
    loss = (y - target).square().mean()
    loss.backward()
    up_grad_first = sum(float(block.up.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.up.weight.grad is not None)
    down_grad_first = sum(float(block.down.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.down.weight.grad is not None)
    opt.step()
    opt.zero_grad(set_to_none=True)
    y2, _ = adapter(x, probs, attn, epoch=0)
    loss2 = (y2 - target).square().mean()
    loss2.backward()
    down_grad = sum(float(block.down.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.down.weight.grad is not None)
    depthwise_grad = sum(float(block.depthwise.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.depthwise.weight.grad is not None)
    up_grad = sum(float(block.up.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.up.weight.grad is not None)
    passed = up_grad_first > 0 and down_grad_first == 0.0 and down_grad > 0 and depthwise_grad > 0 and up_grad > 0
    return {
        "pass": passed,
        "up_grad_first": up_grad_first,
        "down_grad_first": down_grad_first,
        "down_grad_second": down_grad,
        "depthwise_grad_second": depthwise_grad,
        "up_grad_second": up_grad,
        "check": "zero_up_non_dead_startup_gradient",
    }


def _gate_c1(device: torch.device, samples: int = 128) -> dict:
    torch.manual_seed(17)
    adapter = ACPRPredicateAnchoredVisualAdapter(
        dim=128,
        rank=16,
        num_layers=3,
        num_predicates=8,
        grid_hw=(16, 20),
        schedule=VistaScaleSchedule(early_scale=0.15, main_scale=0.15, late_scale=0.08),
    ).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.08, weight_decay=0.0)
    batch_size = 2
    steps = max(1, samples // batch_size) * 8

    def batch(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(batch_size, 3, 320, 128, generator=gen, device=device)
        probs = torch.ones(batch_size, 8, device=device)
        attn = torch.full((batch_size, 8, 320), 1.0 / 320.0, device=device)
        # Local shift residual is representable by the low-rank + depthwise path.
        target = x + 0.08 * torch.tanh(x.roll(shifts=1, dims=2))
        return x, probs, attn, target

    fixed_batches = [batch(100 + step) for step in range(max(1, samples // batch_size))]

    def fixed_loss() -> float:
        vals = []
        with torch.no_grad():
            for x_eval, p_eval, a_eval, t_eval in fixed_batches:
                y_eval, _ = adapter(x_eval, p_eval, a_eval, epoch=0)
                vals.append((y_eval - t_eval).square().sum(dim=-1).mean())
        return float(torch.stack(vals).mean().detach().cpu())

    initial = fixed_loss()
    for step in range(steps):
        x, probs, attn, target = fixed_batches[step % len(fixed_batches)]
        y, _ = adapter(x, probs, attn, epoch=0)
        loss = (y - target).square().sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = fixed_loss()
    return {
        "pass": final < initial * 0.90,
        "initial_loss": initial,
        "final_loss": final,
        "relative_final": final / max(initial, 1e-12),
        "samples": samples,
        "check": "synthetic_adapter_capacity_c1",
    }


def _gate_c(device: torch.device, samples: int = 128) -> dict:
    return _gate_c1(device, samples=samples)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _task_loss(
    model: ACPROIAModel,
    batch: dict[str, Any],
    target_builder: WeakPredicateTargetBuilder,
    matrices: tuple[torch.Tensor, torch.Tensor],
    epoch: int,
    use_final_logits: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(batch["image"], epoch=epoch)
    pred_batch = target_builder.build(batch["file_name"], device=batch["image"].device)
    contradiction = out.get("predicate_reason_contradiction_score_by_label")
    action_logits = out["action_logits_final_raw"] if use_final_logits else out["action_logits_base"]
    reason_logits = out["reason_logits_final_raw"] if use_final_logits else out["reason_logits_base"]
    action = L.action_asl_loss(action_logits, batch["action"])
    reason_partial = L.partial_label_reason_loss(reason_logits, batch["reason"], contradiction)
    reason_f1 = L.reason_soft_f1_loss(reason_logits, batch["reason"], contradiction_scores=contradiction)
    predicate = L.predicate_weak_bce_mil_loss(
        out["predicate_logits"],
        pred_batch["predicate_targets"],
        pred_batch["predicate_mask"],
        pred_batch.get("predicate_reliability"),
    )
    total = action + reason_partial + 0.08 * reason_f1 + 0.12 * predicate
    return total, {
        "action": float(action.detach().cpu()),
        "reason": float((reason_partial + 0.08 * reason_f1).detach().cpu()),
        "predicate": float(predicate.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def _average_losses(model: ACPROIAModel, loader, target_builder, matrices, device: torch.device, epoch: int, max_batches: int) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {"action": 0.0, "reason": 0.0, "predicate": 0.0, "total": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            _, parts = _task_loss(model, batch, target_builder, matrices, epoch)
            for key in sums:
                sums[key] += parts[key]
            count += 1
            if count >= max_batches:
                break
    return {key: val / max(count, 1) for key, val in sums.items()}


def _train_steps(
    model: ACPROIAModel,
    loader,
    target_builder,
    matrices,
    device: torch.device,
    epoch: int,
    steps: int,
    lr: float,
    train_adapter_only: bool = False,
    use_final_logits: bool = False,
) -> dict[str, float]:
    model.train()
    if train_adapter_only:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.visual_adapter.parameters():
            p.requires_grad_(True)
        for p in model.threshold_head.parameters():
            p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    latest: dict[str, float] = {}
    for step, batch in zip(range(steps), cycle(loader)):
        batch = _batch_to_device(batch, device)
        loss, parts = _task_loss(model, batch, target_builder, matrices, epoch, use_final_logits=use_final_logits)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        latest = parts
    return latest


def _runtime_from_config(config_path: str, device: torch.device, batch_size: int, max_samples: int):
    cfg = load_config(config_path)
    loader = make_loader(cfg, "train", batch_size=batch_size, max_samples=max_samples, shuffle=False, num_workers=0)
    grammar = ACPRReasonGrammar(cfg.get("grammar", {}).get("path", "configs/acpr_reason_predicate_grammar.yaml"))
    target_builder = WeakPredicateTargetBuilder(cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml"), cfg.get("bdd100k_root"))
    temp_model = build_model(cfg, device)
    matrices = reason_predicate_matrices(grammar, temp_model.predicate_head.names, device)
    del temp_model
    return cfg, loader, target_builder, matrices


def _gate_c2(config_path: str, device: torch.device, samples: int = 128, steps: int = 8) -> dict:
    torch.manual_seed(19)
    cfg, loader, target_builder, matrices = _runtime_from_config(config_path, device, batch_size=2, max_samples=samples)
    disabled = build_model(cfg, device)
    enabled = build_model(cfg, device)
    enabled.load_state_dict(disabled.state_dict(), strict=False)
    disabled.vista_enabled = False
    enabled.vista_enabled = True
    init_disabled = _average_losses(disabled, loader, target_builder, matrices, device, epoch=0, max_batches=4)
    init_enabled = _average_losses(enabled, loader, target_builder, matrices, device, epoch=0, max_batches=4)
    _train_steps(disabled, loader, target_builder, matrices, device, epoch=0, steps=steps, lr=2e-4)
    _train_steps(enabled, loader, target_builder, matrices, device, epoch=0, steps=steps, lr=2e-4)
    final_disabled = _average_losses(disabled, loader, target_builder, matrices, device, epoch=0, max_batches=4)
    final_enabled = _average_losses(enabled, loader, target_builder, matrices, device, epoch=0, max_batches=4)
    verdicts: dict[str, bool] = {}
    rates: dict[str, dict[str, float]] = {}
    for key in ("action", "reason", "predicate"):
        base_rate = (init_disabled[key] - final_disabled[key]) / max(abs(init_disabled[key]), 1e-8)
        vista_rate = (init_enabled[key] - final_enabled[key]) / max(abs(init_enabled[key]), 1e-8)
        rates[key] = {"disabled_decrease_rate": base_rate, "vista_decrease_rate": vista_rate}
        verdicts[key] = vista_rate >= base_rate
    better_count = sum(bool(v) for v in verdicts.values())
    worst_ratio = min(
        rates[key]["vista_decrease_rate"] / max(rates[key]["disabled_decrease_rate"], 1e-8)
        for key in ("action", "reason", "predicate")
    )
    with torch.no_grad():
        batch = _batch_to_device(next(iter(loader)), device)
        out = enabled(batch["image"], epoch=0)
        delta = out.get("vista_delta_map")
        delta_norm = float(delta.abs().mean().detach().cpu()) if torch.is_tensor(delta) else 0.0
        uniformity = float(out.get("vista_delta_uniformity", torch.tensor(0.0)).detach().cpu()) if torch.is_tensor(out.get("vista_delta_uniformity")) else 0.0
        alpha = float(out.get("vista_alpha_abs_mean", torch.tensor(0.0)).detach().cpu()) if torch.is_tensor(out.get("vista_alpha_abs_mean")) else 0.0
    passed = better_count >= 2 and worst_ratio >= 0.95 and delta_norm > 0 and alpha <= 0.05
    return {
        "pass": bool(passed),
        "samples": samples,
        "steps": steps,
        "initial_disabled": init_disabled,
        "initial_vista": init_enabled,
        "final_disabled": final_disabled,
        "final_vista": final_enabled,
        "loss_decrease_rates": rates,
        "better_count": better_count,
        "worst_rate_ratio": worst_ratio,
        "adapter_delta_norm": delta_norm,
        "adapter_delta_uniformity": uniformity,
        "vista_alpha_abs_mean": alpha,
        "check": "real_bdd_oia_128_sample_mechanism_overfit_c2",
    }


def _collect_metrics(model: ACPROIAModel, loader, device: torch.device, epoch: int, max_batches: int = 8) -> dict:
    model.eval()
    action_logits, reason_logits, action_labels, reason_labels = [], [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = _batch_to_device(batch, device)
            out = model(batch["image"], epoch=epoch)
            action_logits.append(out["action_logits_final_raw"].detach().cpu())
            reason_logits.append(out["reason_logits_final_raw"].detach().cpu())
            action_labels.append(batch["action"].detach().cpu())
            reason_labels.append(batch["reason"].detach().cpu())
            if idx + 1 >= max_batches:
                break
    views = acpr_metric_views(torch.cat(action_logits), torch.cat(reason_logits), torch.cat(action_labels), torch.cat(reason_labels))
    return views["metrics_raw_fixed"]


def _gate_d(config_path: str = "", reference_checkpoint: str = "", device: torch.device | None = None, samples: int = 128, steps: int = 16) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config_path or not Path(config_path).exists():
        return {
            "pass": False,
            "reference_checkpoint": str(reference_checkpoint or config_path),
            "checkpoint_exists": bool(reference_checkpoint and Path(reference_checkpoint).exists()),
            "reason": "config_missing_for_real_train_calib_sanity",
            "required_before_full_train": True,
        }
    path = Path(reference_checkpoint) if reference_checkpoint else None
    if not path or not path.exists():
        return {"pass": False, "reference_checkpoint": str(path) if path else "", "checkpoint_exists": False, "reason": "missing_reference_checkpoint"}
    cfg = load_config(config_path)
    train_ds = make_dataset(cfg, "train")
    main_idx, calib_idx = make_train_calib_indices(
        train_ds,
        calib_fraction=float(cfg.get("threshold", {}).get("train_calib_fraction", 0.10)),
        seed=int(cfg.get("threshold", {}).get("split_seed", 20260615)),
    )
    loader = make_loader(cfg, "train", batch_size=2, max_samples=None, shuffle=False, num_workers=0, indices=list(calib_idx)[:samples])
    grammar = ACPRReasonGrammar(cfg.get("grammar", {}).get("path", "configs/acpr_reason_predicate_grammar.yaml"))
    target_builder = WeakPredicateTargetBuilder(cfg.get("predicate", {}).get("scene_config", "configs/acpr_scene_predicates.yaml"), cfg.get("bdd100k_root"))
    model = build_model(cfg, device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.vista_enabled = True
    matrices = reason_predicate_matrices(grammar, model.predicate_head.names, device)
    before = _collect_metrics(model, loader, device, epoch=0)
    _train_steps(
        model,
        loader,
        target_builder,
        matrices,
        device,
        epoch=0,
        steps=steps,
        lr=3e-4,
        train_adapter_only=True,
        use_final_logits=True,
    )
    # Gate D is specifically adapter + CalAlign/threshold sanity on train_calib.
    # The candidate is derived only from train_calib and copied to deploy params
    # before measuring whether the strong checkpoint can move without damage.
    teacher = collect_threshold_teacher(model, loader, device, epoch=0, cfg=cfg)
    current_theta = model.threshold_head.compose_theta().detach().clone()
    teacher_theta = teacher["threshold_logit"].to(device).detach().clone()
    # Gate D is action-safe: threshold refresh may improve reasons but cannot
    # move action thresholds and mask a visual-action regression.
    teacher_theta[: model.action_dim] = current_theta[: model.action_dim]
    model.threshold_head.update_teacher(
        teacher_theta,
        pred_rate_teacher=teacher["pred_rate"].to(device),
        ema=1.0,
        copy_to_params=True,
    )
    after = _collect_metrics(model, loader, device, epoch=0)
    before_action = float(before.get("Act_mF1", 0.0))
    before_exp = float(before.get("Exp_mF1", 0.0))
    after_action = float(after.get("Act_mF1", 0.0))
    after_exp = float(after.get("Exp_mF1", 0.0))
    before_per_action = before.get("per_action_F1", [])
    after_per_action = after.get("per_action_F1", [])
    per_action_improved = any(float(a) > float(b) for a, b in zip(after_per_action, before_per_action)) if isinstance(before_per_action, list) else False
    passed = after_action >= before_action - 0.001 and after_exp >= before_exp - 0.002 and per_action_improved
    return {
        "pass": bool(passed),
        "reference_checkpoint": str(path),
        "checkpoint_exists": True,
        "samples": samples,
        "steps": steps,
        "baseline": before,
        "after_sanity": after,
        "action_delta": after_action - before_action,
        "exp_delta": after_exp - before_exp,
        "per_action_improved": per_action_improved,
        "action_threshold_preserved": True,
        "check": "strong_checkpoint_train_calib_one_epoch_sanity_d",
    }


def _lane_vertices(poly: Any) -> list[tuple[float, float]]:
    if isinstance(poly, dict):
        vertices = poly.get("vertices") or poly.get("verts") or poly.get("points")
    else:
        vertices = poly
    out: list[tuple[float, float]] = []
    if not isinstance(vertices, list):
        return out
    for item in vertices:
        if isinstance(item, dict) and "x" in item and "y" in item:
            out.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((float(item[0]), float(item[1])))
    return out


def _lane_objects_to_mask(
    objects: list[dict[str, Any]],
    image_size: tuple[int, int],
    output_size: tuple[int, int],
) -> torch.Tensor:
    width, height = image_size
    out_h, out_w = output_size
    mask_img = Image.new("L", (out_w, out_h), 0)
    draw = ImageDraw.Draw(mask_img)
    line_width = max(1, int(round(min(out_h, out_w) / 45)))
    for obj in objects:
        poly_entries = obj.get("poly2d") or []
        if (
            isinstance(poly_entries, list)
            and poly_entries
            and all(isinstance(v, (list, tuple)) and len(v) >= 2 and isinstance(v[0], (int, float)) and isinstance(v[1], (int, float)) for v in poly_entries)
        ):
            poly_entries = [poly_entries]
        elif not isinstance(poly_entries, list):
            poly_entries = [poly_entries]
        for poly in poly_entries:
            vertices = _lane_vertices(poly)
            if len(vertices) < 2:
                continue
            scaled = [
                (
                    max(0, min(out_w - 1, int(round(x / max(width, 1) * out_w)))),
                    max(0, min(out_h - 1, int(round(y / max(height, 1) * out_h)))),
                )
                for x, y in vertices
            ]
            draw.line(scaled, fill=1, width=line_width)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(mask_img.tobytes()))
    return data.view(out_h, out_w).float()


def _masks_for_file(index: BDD100KGroundingIndex, file_name: str, output_size: tuple[int, int] = (45, 80)) -> dict[str, torch.Tensor]:
    paths = index.lookup(file_name)
    object_mask = torch.zeros(output_size, dtype=torch.float32)
    lane_mask = torch.zeros(output_size, dtype=torch.float32)
    drivable_mask = torch.zeros(output_size, dtype=torch.float32)
    if paths.label_json:
        objects = load_bdd100k_objects(paths.label_json)
        object_mask = torch.maximum(
            object_mask,
            objects_to_mask(
                objects,
                image_size=(1280, 720),
                output_size=output_size,
                include_lane=False,
                include_drivable=False,
            ),
        )
        lane_objects = []
        for obj in objects:
            cat = str(obj.get("category", ""))
            if cat.startswith("lane/"):
                lane_objects.append(obj)
        if lane_objects:
            lane_mask = torch.maximum(
                lane_mask,
                _lane_objects_to_mask(lane_objects, image_size=(1280, 720), output_size=output_size),
            )
        drivable_objects = [obj for obj in objects if str(obj.get("category", "")).startswith("area/")]
        if drivable_objects:
            drivable_mask = torch.maximum(
                drivable_mask,
                objects_to_mask(
                    drivable_objects,
                    image_size=(1280, 720),
                    output_size=output_size,
                    include_lane=False,
                    include_drivable=True,
                ),
            )
    if paths.drivable_map:
        drivable_mask = torch.maximum(
            drivable_mask,
            drivable_map_to_mask(paths.drivable_map, output_size=output_size, positive_values={1, 2, 29, 76, 255}),
        )
    combined = torch.maximum(torch.maximum(object_mask, lane_mask), drivable_mask).clamp(0, 1)
    return {
        "combined": combined,
        "object": object_mask.clamp(0, 1),
        "lane": lane_mask.clamp(0, 1),
        "drivable": drivable_mask.clamp(0, 1),
    }


def _mask_for_file(index: BDD100KGroundingIndex, file_name: str, output_size: tuple[int, int] = (45, 80)) -> torch.Tensor:
    return _masks_for_file(index, file_name, output_size=output_size)["combined"]


def _gate_e(config_path: str, device: torch.device | None = None, reference_checkpoint: str = "", samples: int = 32, steps: int = 4) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(config_path).exists():
        return {"pass": False, "bdd100k_root_configured": False, "reason": "config_missing_for_real_localization"}
    cfg = load_config(config_path)
    root = cfg.get("bdd100k_root")
    if not root:
        return {"pass": False, "bdd100k_root_configured": False, "reason": "missing_bdd100k_root"}
    if "data_root" not in cfg or "raw_root" not in cfg:
        return {
            "pass": False,
            "bdd100k_root_configured": True,
            "reason": "missing_data_root_or_raw_root_for_real_localization",
            "required_before_full_train": True,
        }
    cfg, loader, target_builder, matrices = _runtime_from_config(config_path, device, batch_size=2, max_samples=samples)
    model = build_model(cfg, device)
    if reference_checkpoint and Path(reference_checkpoint).exists():
        ckpt = torch.load(reference_checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.vista_enabled = True
    _train_steps(model, loader, target_builder, matrices, device, epoch=0, steps=steps, lr=3e-4, train_adapter_only=True)
    index = BDD100KGroundingIndex(root)
    grounded_mass_values: list[float] = []
    random_expected_values: list[float] = []
    object_mass_values: list[float] = []
    lane_mass_values: list[float] = []
    drivable_mass_values: list[float] = []
    used = 0
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            out = model(batch["image"], epoch=6)
            delta = out.get("vista_delta_map")
            if not torch.is_tensor(delta):
                continue
            sample_delta = delta.abs().mean(dim=1).view(delta.shape[0], 45, 80).detach().cpu()
            for i, fn in enumerate(batch["file_name"]):
                masks = _masks_for_file(index, str(fn), output_size=(45, 80))
                mask = masks["combined"]
                if mask.sum() < 1:
                    continue
                dm = sample_delta[i]
                total = dm.sum().clamp_min(1e-8)
                grounded_fraction = float((dm * mask).sum() / total)
                random_expected = float(mask.mean())
                grounded_mass_values.append(grounded_fraction)
                random_expected_values.append(random_expected)
                if masks["object"].sum() > 0:
                    object_mass_values.append(float((dm * masks["object"]).sum() / total))
                if masks["lane"].sum() > 0:
                    lane_mass_values.append(float((dm * masks["lane"]).sum() / total))
                if masks["drivable"].sum() > 0:
                    drivable_mass_values.append(float((dm * masks["drivable"]).sum() / total))
                used += 1
            if used >= samples:
                break
    if not grounded_mass_values:
        return {"pass": False, "bdd100k_root_configured": True, "used_samples": 0, "reason": "no_grounded_masks_available"}
    grounded_mean = sum(grounded_mass_values) / len(grounded_mass_values)
    random_mean = sum(random_expected_values) / len(random_expected_values)
    margin = grounded_mean - random_mean
    return {
        "pass": margin > 0.01,
        "bdd100k_root_configured": True,
        "used_samples": used,
        "grounded_delta_mass_mean": grounded_mean,
        "random_equal_area_mass_mean": random_mean,
        "object_delta_mass_mean": sum(object_mass_values) / len(object_mass_values) if object_mass_values else 0.0,
        "lane_delta_mass_mean": sum(lane_mass_values) / len(lane_mass_values) if lane_mass_values else 0.0,
        "drivable_delta_mass_mean": sum(drivable_mass_values) / len(drivable_mass_values) if drivable_mass_values else 0.0,
        "object_mask_samples": len(object_mass_values),
        "lane_mask_samples": len(lane_mass_values),
        "drivable_mask_samples": len(drivable_mass_values),
        "localization_margin": margin,
        "check": "bdd100k_object_drivable_lane_delta_localization_e",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mechanism_samples", type=int, default=128)
    parser.add_argument("--mechanism_steps", type=int, default=8)
    parser.add_argument("--sanity_samples", type=int, default=128)
    parser.add_argument("--sanity_steps", type=int, default=16)
    parser.add_argument("--localization_samples", type=int, default=32)
    parser.add_argument("--localization_steps", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate_specs = [
        ("VISTA_GATE_A_EQUIVALENCE.json", lambda: _gate_a(device)),
        ("VISTA_GATE_B_GRADIENT.json", lambda: _gate_b(device)),
        ("VISTA_GATE_C1_SYNTHETIC_CAPACITY.json", lambda: _gate_c1(device, samples=args.mechanism_samples)),
        ("VISTA_GATE_C2_REAL_128_MECHANISM.json", lambda: _gate_c2(args.config, device, samples=args.mechanism_samples, steps=args.mechanism_steps)),
        ("VISTA_GATE_D_TRAIN_CALIB_SANITY.json", lambda: _gate_d(args.config, args.reference_checkpoint, device, samples=args.sanity_samples, steps=args.sanity_steps)),
        ("VISTA_GATE_E_LOCALIZATION.json", lambda: _gate_e(args.config, device, reference_checkpoint=args.reference_checkpoint, samples=args.localization_samples, steps=args.localization_steps)),
    ]
    gate_payloads = {}
    for name, fn in gate_specs:
        try:
            payload = fn()
        except Exception as exc:
            payload = {"pass": False, "error": type(exc).__name__, "message": str(exc), "gate_exception": True}
        gate_payloads[name] = payload
        _json_dump(out / name, payload)
    passed = all(bool(payload.get("pass")) for payload in gate_payloads.values())
    summary = {
        "pass": passed,
        "files": list(gate_payloads),
        "blocking_failures": [k for k, v in gate_payloads.items() if not v.get("pass")],
    }
    _json_dump(out / "VISTA_GATES_PASS.json", summary)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
