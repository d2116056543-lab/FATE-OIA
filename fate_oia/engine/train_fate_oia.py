from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

import utils
import vision_transformer as vits
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.datasets.bdd100k_grounding import load_bdd100k_objects
from fate_oia.grounding.losses import attention_grounding_bce
from fate_oia.grounding.mask_builder import objects_to_mask
from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.token_provenance import keep_merge_tokens, recover_attribution


def build_transform(image_height: int, image_width: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def build_backbone(args, device: torch.device) -> tuple[nn.Module, int]:
    if args.arch not in vits.__dict__:
        raise ValueError(f"Unsupported SNNA ViT arch: {args.arch}")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, int(model.embed_dim)


@torch.no_grad()
def extract_tokens(backbone: nn.Module, images: torch.Tensor, n_last_blocks: int) -> torch.Tensor:
    layers = backbone.get_intermediate_layers(images, n_last_blocks)
    return layers[-1]


def labels_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([batch["action"].float(), batch["reason"].float()], dim=1)


def limited(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def make_loader(args, split: str, shuffle: bool) -> DataLoader:
    ds = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=True,
        transform=build_transform(args.image_height, args.image_width),
    )
    max_samples = args.max_train_samples if split == "train" else args.max_val_samples
    ds = limited(ds, max_samples)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def make_multilabel_criterion(args) -> nn.Module:
    if args.loss == "asl":
        return AsymmetricLossMultiLabel(gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg, clip=args.asl_clip)
    return nn.BCEWithLogitsLoss()
def load_grounding_cache(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fn = rec.get("file_name")
            if fn:
                out[str(fn)] = rec
    return out


def compute_grounding_loss(
    label_attention: torch.Tensor | None,
    batch: dict[str, Any],
    grounding_cache: dict[str, dict[str, Any]],
    args,
    device: torch.device,
) -> torch.Tensor:
    if label_attention is None or not grounding_cache:
        return torch.zeros((), device=device)
    attn_maps = attention_to_patch_map(label_attention, slice(0, args.action_dim + args.reason_dim), args.image_height, args.image_width, args.patch_size)
    losses = []
    categories = set(x.strip() for x in args.grounding_categories.split(",") if x.strip()) if args.grounding_categories else None
    file_names = batch.get("file_name", [])
    if isinstance(file_names, str):
        file_names = [file_names]
    for i, fn in enumerate(file_names):
        rec = grounding_cache.get(str(fn))
        if not rec or not rec.get("label_json"):
            continue
        try:
            objects = load_bdd100k_objects(rec["label_json"])
            target = objects_to_mask(objects, (args.grounding_image_width, args.grounding_image_height), attn_maps.shape[-2:], categories=categories).to(device)
        except Exception:
            continue
        if float(target.sum().item()) <= 0:
            continue
        losses.append(attention_grounding_bce(attn_maps[i], target))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()



def compress_tokens(tokens: torch.Tensor, keep_ratio: float, num_summary_tokens: int, min_tokens: int) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if keep_ratio >= 0.999:
        return tokens, None, {"enabled": False, "original_tokens": int(tokens.shape[1]), "reduced_tokens": int(tokens.shape[1])}
    cls = tokens[:, :1]
    patch = tokens[:, 1:]
    scores = patch.norm(dim=-1)
    reduced_patch, patch_prov, stats = keep_merge_tokens(patch, scores=scores, keep_ratio=keep_ratio, num_summary_tokens=num_summary_tokens, min_tokens=min_tokens)
    reduced = torch.cat([cls, reduced_patch], dim=1)
    b, n, _ = tokens.shape
    r = reduced.shape[1]
    prov = torch.zeros(b, n, r, device=tokens.device, dtype=tokens.dtype)
    prov[:, 0, 0] = 1.0
    prov[:, 1:, 1:] = patch_prov
    stats = {**stats, "enabled": True, "original_tokens": int(n), "reduced_tokens": int(r)}
    return reduced, prov, stats


def recover_label_attention(attention: torch.Tensor | None, provenance: torch.Tensor | None, original_tokens: int) -> torch.Tensor | None:
    if attention is None:
        return None
    # label-query head returns [B,H,L,N]. Average heads -> [B,L,N].
    if attention.ndim == 4:
        attn = attention.mean(1)
    elif attention.ndim == 3:
        attn = attention
    else:
        raise ValueError(f"Unexpected attention shape {tuple(attention.shape)}")
    if provenance is None:
        return attn
    recovered = recover_attribution(attn.transpose(1, 2), provenance).transpose(1, 2)
    if recovered.shape[-1] != original_tokens:
        raise ValueError("Recovered attention length mismatch")
    return recovered


def attention_to_patch_map(label_attention: torch.Tensor, label_slice: slice, image_height: int, image_width: int, patch_size: int) -> torch.Tensor:
    # label_attention: [B,L,N_original]. Drop CLS and average selected labels.
    patch_scores = label_attention[:, label_slice, 1:].mean(1)
    h = image_height // patch_size
    w = image_width // patch_size
    if patch_scores.shape[1] != h * w:
        side = int(math.sqrt(patch_scores.shape[1]))
        if side * side != patch_scores.shape[1]:
            raise ValueError(f"Cannot reshape {patch_scores.shape[1]} patch tokens into a map")
        h = w = side
    return patch_scores.reshape(patch_scores.shape[0], h, w)


def reason_to_action_consistency_loss(action_logits: torch.Tensor, r2a_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(r2a_logits, torch.sigmoid(action_logits).detach())


def counterfactual_deletion_loss(model: nn.Module, tokens: torch.Tensor, labels: torch.Tensor, base_loss: torch.Tensor, action_dim: int, topk_ratio: float = 0.15, margin: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        out = model(tokens)
        attn = out.get("attention")
        if attn is None:
            return tokens.new_zeros(())
        attn_mean = attn.mean(1).mean(1) if attn.ndim == 4 else attn.mean(1)
        patch_scores = attn_mean[:, 1:] if attn_mean.shape[1] == tokens.shape[1] else attn_mean
        k = max(1, int(round(patch_scores.shape[1] * topk_ratio)))
        topk = torch.topk(patch_scores, k=k, dim=1).indices + 1
    masked = tokens.clone()
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device).unsqueeze(1)
    masked[batch_idx, topk] = 0.0
    masked_out = model(masked)
    masked_logits = torch.cat([masked_out["action_logits"], masked_out["reason_logits"]], dim=1)
    masked_loss = F.binary_cross_entropy_with_logits(masked_logits, labels.float())
    return F.relu(margin + base_loss.detach() - masked_loss)


def run_epoch(args, backbone, model, loader, criterion, optimizer, device, train: bool, grounding_cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    model.train(train)
    total_loss = 0.0
    count = 0
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    stats_rows: list[dict[str, Any]] = []
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = labels_from_batch(batch).to(device, non_blocking=True)
        with torch.no_grad():
            original_tokens = extract_tokens(backbone, images, args.n_last_blocks)
        tokens, provenance, token_stats = compress_tokens(original_tokens, args.token_keep_ratio, args.num_summary_tokens, args.min_tokens)
        out = model(tokens)
        logits = torch.cat([out["action_logits"], out["reason_logits"]], dim=1)
        main_loss = criterion(logits, labels)
        r2a_loss = reason_to_action_consistency_loss(out["action_logits"], out["reason_to_action_logits"])
        loss = main_loss + args.loss_reason_to_action * r2a_loss
        cf_loss = original_tokens.new_zeros(())
        if args.loss_counterfactual > 0:
            cf_loss = counterfactual_deletion_loss(model, tokens, labels, main_loss, args.action_dim, args.counterfactual_topk_ratio)
            loss = loss + args.loss_counterfactual * cf_loss
        grounding_loss = original_tokens.new_zeros(())
        if args.loss_grounding > 0 and grounding_cache:
            recovered_attn = recover_label_attention(out.get("attention"), provenance, original_tokens.shape[1])
            grounding_loss = compute_grounding_loss(recovered_attn, batch, grounding_cache, args, device)
            loss = loss + args.loss_grounding * grounding_loss
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = images.shape[0]
        total_loss += float(loss.item()) * bs
        count += bs
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        if len(stats_rows) < args.max_saved_token_stats:
            stats_rows.append({k: (int(v) if isinstance(v, int) else v) for k, v in token_stats.items()})
        if step % args.log_every == 0:
            print(json.dumps({
                "event": "fate_oia_batch",
                "train": train,
                "step": step,
                "loss": float(loss.item()),
                "main_loss": float(main_loss.item()),
                "r2a_loss": float(r2a_loss.item()),
                "cf_loss": float(cf_loss.item()),
                "grounding_loss": float(grounding_loss.item()) if "grounding_loss" in locals() else 0.0,
                "batch_size": bs,
                "token_stats": token_stats,
            }), flush=True)
    logits_tensor = torch.cat(logits_all, 0) if logits_all else torch.empty(0, args.action_dim + args.reason_dim)
    labels_tensor = torch.cat(labels_all, 0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    metrics = evaluate_snna25(logits_tensor, labels_tensor, args.action_dim, threshold_mode=args.threshold_mode, fixed_threshold=args.eval_threshold)["metrics"]
    return {"loss": total_loss / max(count, 1), "count": count, "metrics": metrics, "logits": logits_tensor, "labels": labels_tensor, "token_stats": stats_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train full FATE-OIA token model with label-query, reason-to-action, optional compression and counterfactual loss.")
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--action_dim", type=int, default=4, choices=[4, 5])
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=224)
    ap.add_argument("--image_width", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["bce", "asl"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_reason_to_action", type=float, default=0.1)
    ap.add_argument("--loss_counterfactual", type=float, default=0.0)
    ap.add_argument("--loss_grounding", type=float, default=0.0)
    ap.add_argument("--grounding_cache_jsonl", default="")
    ap.add_argument("--grounding_categories", default="person,rider,bike,car,bus,truck,motor,traffic light,traffic sign,lane/crosswalk")
    ap.add_argument("--grounding_image_width", type=int, default=1280)
    ap.add_argument("--grounding_image_height", type=int, default=720)
    ap.add_argument("--counterfactual_topk_ratio", type=float, default=0.15)
    ap.add_argument("--token_keep_ratio", type=float, default=1.0)
    ap.add_argument("--num_summary_tokens", type=int, default=4)
    ap.add_argument("--min_tokens", type=int, default=16)
    ap.add_argument("--threshold_mode", choices=["fixed", "global", "per_label"], default="fixed")
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_saved_token_stats", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device)
    model = FATEOIAFeatureModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim, use_label_query=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_multilabel_criterion(args)
    train_loader = make_loader(args, "train", True)
    val_loader = make_loader(args, "val", False)
    grounding_cache = load_grounding_cache(args.grounding_cache_jsonl) if args.grounding_cache_jsonl else {}
    if args.loss_grounding > 0 and not grounding_cache:
        print(json.dumps({"event": "fate_oia_grounding_disabled", "reason": "empty_grounding_cache"}), flush=True)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        train_stats = run_epoch(args, backbone, model, train_loader, criterion, optimizer, device, True, grounding_cache)
        val_stats = run_epoch(args, backbone, model, val_loader, criterion, optimizer, device, False, grounding_cache)
        score = float(val_stats["metrics"].get("Act_mF1", 0.0)) + float(val_stats["metrics"].get("Exp_mF1", 0.0))
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "val_loss": val_stats["loss"], "val_metrics": val_stats["metrics"], "selection_score": score}
        history.append(row)
        with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        latest = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "dim": dim, "best_score": max(best, score)}
        torch.save(latest, out_dir / "checkpoint_latest.pth")
        torch.save(val_stats["logits"], out_dir / "val_logits_latest.pt")
        torch.save(val_stats["labels"], out_dir / "val_labels_latest.pt")
        (out_dir / "token_stats_latest.json").write_text(json.dumps({"train": train_stats["token_stats"], "val": val_stats["token_stats"]}, indent=2), encoding="utf-8")
        if score >= best:
            best = score
            torch.save(latest, out_dir / "checkpoint_best.pth")
            torch.save(val_stats["logits"], out_dir / "val_logits_best.pt")
            torch.save(val_stats["labels"], out_dir / "val_labels_best.pt")
        print(json.dumps({"event": "fate_oia_epoch", **row}), flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()