from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.models.acpr_visual_token_adapter import ACPRPredicateAnchoredVisualAdapter


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
        "vista_gate_mean": float(stats["vista_gate_mean"].detach().cpu()),
        "check": "zero_gate_equivalence_without_bypass",
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
    gate_grad_first = float(adapter.gate_raw.grad.abs().sum().detach().cpu())
    opt.step()
    gate_abs_after_step = float(adapter.gate_raw.detach().abs().sum().cpu())
    opt.zero_grad(set_to_none=True)
    y2, _ = adapter(x, probs, attn, epoch=0)
    loss2 = (y2 - target).square().mean()
    loss2.backward()
    down_grad = sum(float(block.down.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.down.weight.grad is not None)
    depthwise_grad = sum(float(block.depthwise.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.depthwise.weight.grad is not None)
    up_grad = sum(float(block.up.weight.grad.abs().sum().detach().cpu()) for block in adapter.blocks if block.up.weight.grad is not None)
    passed = gate_grad_first > 0 and gate_abs_after_step > 0 and down_grad > 0 and depthwise_grad > 0 and up_grad > 0
    return {
        "pass": passed,
        "gate_grad_first": gate_grad_first,
        "gate_abs_after_step": gate_abs_after_step,
        "down_grad_second": down_grad,
        "depthwise_grad_second": depthwise_grad,
        "up_grad_second": up_grad,
        "check": "non_dead_startup_gradient",
    }


def _gate_c(device: torch.device, samples: int = 128) -> dict:
    torch.manual_seed(17)
    adapter = ACPRPredicateAnchoredVisualAdapter(
        dim=64,
        rank=8,
        num_layers=3,
        num_predicates=8,
        grid_hw=(8, 8),
    ).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.10, weight_decay=0.0)
    batch_size = 4
    steps = max(1, samples // batch_size)

    def batch(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(batch_size, 3, 64, 64, generator=gen, device=device)
        probs = torch.rand(batch_size, 8, generator=gen, device=device)
        attn = torch.softmax(torch.randn(batch_size, 8, 64, generator=gen, device=device), dim=-1)
        target = x + 0.08 * torch.tanh(x.roll(shifts=1, dims=2))
        return x, probs, attn, target

    with torch.no_grad():
        x0, p0, a0, t0 = batch(100)
        y0, _ = adapter(x0, p0, a0, epoch=3)
        initial = float((y0 - t0).square().sum(dim=-1).mean().detach().cpu())
    for step in range(steps):
        x, probs, attn, target = batch(100 + step)
        y, _ = adapter(x, probs, attn, epoch=3)
        loss = (y - target).square().sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        x1, p1, a1, t1 = batch(100)
        y1, _ = adapter(x1, p1, a1, epoch=3)
        final = float((y1 - t1).square().sum(dim=-1).mean().detach().cpu())
    return {
        "pass": final < initial * 0.90,
        "initial_loss": initial,
        "final_loss": final,
        "relative_final": final / max(initial, 1e-12),
        "samples": samples,
        "check": "synthetic_mechanism_overfit_real_adapter",
    }


def _gate_d(reference_checkpoint: str) -> dict:
    path = Path(reference_checkpoint) if reference_checkpoint else None
    exists = bool(path and path.exists())
    return {
        "pass": False,
        "reference_checkpoint": str(path) if path else "",
        "checkpoint_exists": exists,
        "reason": "train_calib_one_epoch_sanity_not_executed_by_this_gate",
        "required_before_full_train": True,
    }


def _gate_e(config_path: str) -> dict:
    text = Path(config_path).read_text(encoding="utf-8")
    has_bdd = "bdd100k_root" in text
    return {
        "pass": False,
        "bdd100k_root_configured": has_bdd,
        "reason": "real_object_drivable_lane_delta_localization_not_computed",
        "required_before_full_train": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mechanism_samples", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gate_payloads = {
        "VISTA_GATE_A_EQUIVALENCE.json": _gate_a(device),
        "VISTA_GATE_B_GRADIENT.json": _gate_b(device),
        "VISTA_GATE_C_MECHANISM_OVERFIT.json": _gate_c(device, samples=args.mechanism_samples),
        "VISTA_GATE_D_TRAIN_CALIB_SANITY.json": _gate_d(args.reference_checkpoint),
        "VISTA_GATE_E_LOCALIZATION.json": _gate_e(args.config),
    }
    for name, payload in gate_payloads.items():
        (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    passed = all(bool(payload.get("pass")) for payload in gate_payloads.values())
    summary = {"pass": passed, "files": list(gate_payloads), "blocking_failures": [k for k, v in gate_payloads.items() if not v.get("pass")]}
    (out / "VISTA_GATES_PASS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
