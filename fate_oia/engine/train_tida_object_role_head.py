from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd100k_object_roles import BDD100KObjectRoleDataset, ROLE_NAMES
from fate_oia.engine.train_tida_oia import load_config, load_frozen_vetra_base
from fate_oia.models.tida_object_intent_flow import TIDAObjectRoleHead


def _average_precision(scores: torch.Tensor, target: torch.Tensor) -> float:
    order = scores.argsort(descending=True)
    truth = target[order].float()
    positives = truth.sum()
    if positives <= 0:
        return float("nan")
    precision = truth.cumsum(0) / torch.arange(1, truth.numel() + 1, dtype=torch.float32)
    return float((precision * truth).sum() / positives)


@torch.no_grad()
def evaluate(image_base, role_head, loader, device) -> dict:
    role_head.eval()
    probs, targets = [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["role_target"].to(device, non_blocking=True)
        tokens = image_base.encode_images(image)["patch_tokens_last"]
        probs.append(role_head(tokens).softmax(-1).cpu())
        targets.append(target.cpu())
    probability = torch.cat(probs).flatten(0, 1)
    target = torch.cat(targets).flatten()
    prediction = probability.argmax(-1)
    rows = {}
    for index, name in enumerate(ROLE_NAMES):
        positive = target == index
        true_positive = ((prediction == index) & positive).sum().float()
        precision = true_positive / (prediction == index).sum().clamp_min(1)
        recall = true_positive / positive.sum().clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
        rows[name] = {
            "count": int(positive.sum()),
            "ap": _average_precision(probability[:, index], positive),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
    foreground = target > 0
    rows["foreground_macro_ap"] = sum(rows[name]["ap"] for name in ROLE_NAMES[1:]) / 4.0
    rows["foreground_accuracy"] = float(((prediction > 0) == foreground).float().mean())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--label_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(20260825)
    torch.manual_seed(20260825)
    device = torch.device(args.device)
    config = load_config(args.config)
    image_base = load_frozen_vetra_base(
        config, Path(config["image_base"]["checkpoint"]), device
    ).eval()
    dataset = BDD100KObjectRoleDataset(args.image_root, args.label_root)
    indices = list(range(len(dataset)))
    random.Random(20260825).shuffle(indices)
    indices = indices[: min(args.max_images, len(indices))]
    split = max(1, int(0.85 * len(indices)))
    train_set, valid_set = Subset(dataset, indices[:split]), Subset(dataset, indices[split:])
    loader_args = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_args["prefetch_factor"] = 2
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)
    head = TIDAObjectRoleHead(dim=int(config["model"]["dim"]), num_roles=len(ROLE_NAMES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=0.02)
    class_weight = torch.tensor([0.15, 1.5, 3.0, 3.0, 0.35], device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(args.epochs):
        head.train()
        total_loss, batches = 0.0, 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["role_target"].to(device, non_blocking=True)
            with torch.no_grad():
                tokens = image_base.encode_images(image)["patch_tokens_last"]
            logits = head(tokens)
            ce = F.cross_entropy(
                logits.flatten(0, 1), target.flatten(), weight=class_weight, reduction="none"
            ).view_as(target)
            probability = logits.softmax(-1).gather(-1, target[..., None]).squeeze(-1)
            loss = ((1.0 - probability).square() * ce).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        metrics = evaluate(image_base, head, valid_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / max(batches, 1), "validation": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = metrics["foreground_macro_ap"]
        if score > best:
            best = score
            torch.save({
                "role_head": head.state_dict(), "role_names": ROLE_NAMES,
                "epoch": epoch, "validation": metrics,
                "image_count": len(indices), "train_count": len(train_set), "valid_count": len(valid_set),
                "source": "BDD100K official train images/labels only; no BDD-OIA action/reason/test labels",
            }, output_dir / "checkpoint_best_role_head.pth")
    (output_dir / "role_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
