from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fate_oia.datasets.dino_token_cache import DinoTokenCache
from fate_oia.engine.train_fate_oia import build_backbone, extract_tokens, labels_from_batch, make_loader
from fate_oia.engine.train_trace_oia import parse_args as parse_train_args
from fate_oia.utils.trace_artifacts import append_jsonl, json_safe, write_json


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_trace_oia_v1.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--required_hit_rate", type=float, default=0.99)
    return ap


def _merged_args(argv=None):
    pre = build_parser().parse_args(argv)
    train_args = parse_train_args([
        "--config", pre.config,
        "--output_dir", pre.output_dir,
        "--batch_size", str(pre.batch_size),
        "--num_workers", str(pre.num_workers),
        "--device", pre.device,
        "--max_train_samples", str(pre.max_train_samples),
        "--max_test_samples", str(pre.max_test_samples),
    ])
    if pre.cache_dir:
        train_args.cache_dir = pre.cache_dir
    train_args.log_every = pre.log_every
    train_args.feature_cache_required_hit_rate = pre.required_hit_rate
    return train_args


def _cache_split(args, split: str, backbone, cache: DinoTokenCache, device: torch.device, summary: dict):
    loader = make_loader(args, split, False)
    total = 0
    hits = 0
    built = 0
    file_names: list[str] = []
    start = time.time()
    for step, batch in enumerate(loader):
        files = [str(x) for x in batch.get("file_name", [])]
        file_names.extend(files)
        labels = labels_from_batch(batch)
        rows = [cache.get(fn) for fn in files]
        missing = [i for i, row in enumerate(rows) if row is None]
        hits += len(files) - len(missing)
        total += len(files)
        if missing:
            images = batch["image"].to(device)
            with torch.no_grad():
                tokens = extract_tokens(backbone, images, args.n_last_blocks).detach().cpu()
            for i in missing:
                cache.put(files[i], tokens[i], labels[i])
            built += len(missing)
        if step % max(1, args.log_every) == 0:
            event = {"event": "trace_cache_build_batch", "split": split, "step": step, "total_seen": total, "preexisting_hits": hits, "built": built, "cache_hit_rate_observed": hits / max(1, total)}
            print(json.dumps(json_safe(event)), flush=True)
            append_jsonl(Path(args.output_dir) / "cache_build_progress.jsonl", event)
    split_audit = cache.audit(file_names)
    split_summary = {"split": split, "total": total, "preexisting_hits": hits, "built": built, "seconds": time.time() - start, "audit": split_audit}
    summary["splits"].append(split_summary)


def main(argv=None):
    args = _merged_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.feature_cache_enabled:
        raise SystemExit("feature_cache.enabled is false; refusing cache build")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone, _ = build_backbone(args, device)
    cache = DinoTokenCache(args.cache_dir, args.image_height, args.image_width, args.arch, args.patch_size)
    manifest = {"created_by": "build_trace_oia_token_cache", "config": args.config, "cache_dir": args.cache_dir, "image_height": args.image_height, "image_width": args.image_width, "arch": args.arch, "patch_size": args.patch_size, "best_selection_split": "test", "test_only_evaluation": True}
    cache.write_manifest(manifest)
    summary: dict = {"event": "trace_cache_build_summary", "cache_dir": args.cache_dir, "splits": [], "required_hit_rate": args.feature_cache_required_hit_rate}
    for split in ("train", "test"):
        _cache_split(args, split, backbone, cache, device, summary)
    all_files: list[str] = []
    for item in summary["splits"]:
        # The split-level audit already preserves exact counts. Keep aggregate
        # explicit without re-reading loader metadata in the training process.
        all_files.extend([item["split"]] * int(item["total"]))
    checked = sum(int(s["audit"]["checked"]) for s in summary["splits"])
    present = sum(int(s["audit"]["present"]) for s in summary["splits"])
    summary["cache_audit"] = {"checked": checked, "present": present, "cache_hit_rate": float(present / checked) if checked else 1.0, "cache_root": str(cache.root)}
    total = sum(s["total"] for s in summary["splits"])
    built = sum(s["built"] for s in summary["splits"])
    summary["total_requested"] = total
    summary["built_total"] = built
    summary["final_hit_rate_expected"] = float(summary["cache_audit"]["cache_hit_rate"])
    if summary["final_hit_rate_expected"] + 1e-12 < args.feature_cache_required_hit_rate:
        raise RuntimeError(f"Cache hit-rate failed: {summary['final_hit_rate_expected']} < {args.feature_cache_required_hit_rate}")
    write_json(out / "cache_build_summary.json", summary)
    print(json.dumps(json_safe(summary)), flush=True)


if __name__ == "__main__":
    main()
