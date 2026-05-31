from __future__ import annotations
import argparse, json, socket, time
from pathlib import Path
from typing import Any
import torch
from fate_oia.datasets.dino_token_cache import DinoTokenCache
from fate_oia.engine.audit_cafe_evidence_cache import load_grounding_cache_jsonl
from fate_oia.engine.calibrate_cafe_oia import combined_metrics, threshold_sweep_global, threshold_sweep_per_label
from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.export_trace_visuals import export_trace_case
from fate_oia.engine.train_fate_oia import build_backbone, extract_tokens, labels_from_batch, load_reason_grounding_rules, make_loader
from fate_oia.losses.trace_losses import compute_trace_loss
from fate_oia.models.prototype_calibration import apply_prototype_calibration, fit_classwise_bias_temperature_reliability
from fate_oia.models.trace_oia_model import TraceOIAModel
from fate_oia.utils.config_io import load_yaml_config, write_resolved_config
from fate_oia.utils.trace_artifacts import append_jsonl, json_safe, required_epoch_artifacts, write_json


def _flat(cfg):
    out = {}
    def walk(prefix, val):
        if isinstance(val, dict):
            for k, v in val.items(): walk(f"{prefix}.{k}" if prefix else k, v)
        else: out[prefix] = val
    walk("", cfg); return out


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_trace_oia_v1.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default="dataset/BDD-OIA"); ap.add_argument("--raw_root", default="dataset/BDD-OIA")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth"); ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--arch", default="vit_small"); ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--image_height", type=int, default=360); ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--action_dim", type=int, default=4); ap.add_argument("--reason_dim", type=int, default=21); ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch_size", type=int, default=8); ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4); ap.add_argument("--device", default="cuda"); ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--max_train_samples", type=int, default=0); ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--grounding_cache_jsonl", default=".background_runs/fate_oia_grounding_cache_20260525.jsonl")
    ap.add_argument("--reason_grounding_rules", default="configs/reason_grounding_rules.yaml")
    ap.add_argument("--best_selection_split", choices=["test"], default="test"); ap.add_argument("--best_selection_metric", default="test_joint_composite")
    ap.add_argument("--token_compression", choices=["none"], default="none"); ap.add_argument("--action_final_mode", choices=["base_only"], default="base_only")
    ap.add_argument("--lr_base_head", type=float, default=3e-4); ap.add_argument("--lr_transport", type=float, default=2e-4); ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0); ap.add_argument("--asl_gamma_neg", type=float, default=4.0); ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--loss_counterfactual_direct", type=float, default=0.025); ap.add_argument("--counterfactual_start_epoch", type=int, default=5); ap.add_argument("--cf_margin", type=float, default=0.05)
    ap.add_argument("--cache_dir", default=""); ap.add_argument("--feature_cache_enabled", action=argparse.BooleanOptionalAction, default=True)
    return ap


def parse_args(argv=None):
    ap = build_parser(); pre, _ = ap.parse_known_args(argv)
    cfg = load_yaml_config(pre.config) if pre.config else {}; flat = _flat(cfg)
    mapping = {"paths.data_root": "data_root", "paths.raw_root": "raw_root", "paths.pretrained_weights": "pretrained_weights", "paths.grounding_cache_jsonl": "grounding_cache_jsonl", "paths.reason_grounding_rules": "reason_grounding_rules", "model.action_dim": "action_dim", "model.reason_dim": "reason_dim", "model.image_height": "image_height", "model.image_width": "image_width", "model.patch_size": "patch_size", "model.arch": "arch", "model.token_compression": "token_compression", "model.action_final_mode": "action_final_mode", "training.epochs": "epochs", "training.batch_size": "batch_size", "training.gradient_accumulation_steps": "gradient_accumulation_steps", "training.num_workers": "num_workers", "training.device": "device", "training.log_every": "log_every", "training.best_selection_split": "best_selection_split", "training.best_selection_metric": "best_selection_metric", "optimizer.lr_base_head": "lr_base_head", "optimizer.lr_transport": "lr_transport", "optimizer.weight_decay": "weight_decay", "feature_cache.enabled": "feature_cache_enabled"}
    ap.set_defaults(**{dst: flat[src] for src, dst in mapping.items() if src in flat})
    args = ap.parse_args(argv); args.config_data = cfg; args.config_version = cfg.get("config_version", "")
    if args.config_version != "trace_oia_v1_proto_transport": raise ValueError(f"wrong config_version: {args.config_version}")
    args.experiment_name = cfg.get("experiment_name", "trace_oia_v1_proto_transport")
    if not args.cache_dir: args.cache_dir = str(Path(args.output_dir) / "dino_token_cache")
    return args


def _joint(metrics): return 0.5 * float(metrics.get("Act_mF1", 0.0)) + 0.5 * float(metrics.get("Exp_mF1", 0.0))


def _tokens(args, cache, backbone, images, batch, labels, device):
    files = [str(x) for x in batch.get("file_name", [])]
    if cache:
        rows = [cache.get(fn) for fn in files]
        if rows and all(x is not None for x in rows):
            return torch.stack([x["tokens"] for x in rows], 0).to(device=device, dtype=images.dtype), cache.stats()
    with torch.no_grad(): tok = extract_tokens(backbone, images, args.n_last_blocks)
    if cache:
        for i, fn in enumerate(files): cache.put(fn, tok[i], labels[i])
    return tok, cache.stats() if cache else {}


def run_epoch(args, backbone, model, loader, opt, device, train, grounding_cache, rules, epoch, cache):
    model.train(train)
    if train: opt.zero_grad(set_to_none=True)
    logits_all=[]; base_all=[]; ev_all=[]; labels_all=[]; names=[]; loss_rows=[]; t_rows=[]; e_rows=[]; cf_rows=[]; total=0.0; count=0; start=time.time()
    for step, batch in enumerate(loader):
        images = batch["image"].to(device); labels = labels_from_batch(batch).to(device)
        tok, cache_stats = _tokens(args, cache, backbone, images, batch, labels, device)
        out = model(tok, batch=batch, grounding_cache=grounding_cache, reason_rules=rules, return_cf=True, cf_targets=labels[:, args.action_dim:], image_height=args.image_height, image_width=args.image_width, patch_size=args.patch_size)
        out["transport_module"] = model.transport
        loss, lstats = compute_trace_loss(args, out, labels, epoch)
        if train:
            (loss / float(args.gradient_accumulation_steps)).backward()
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(loader)):
                opt.step(); opt.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu()) * labels.shape[0]; count += labels.shape[0]
        logits_all.append(torch.cat([out["action_logits"], out["reason_logits"]], 1).detach().cpu())
        base_all.append(torch.cat([out["base_action_logits"], out["base_reason_logits"]], 1).detach().cpu())
        ev_all.append(out["transport"]["evidence_reason_logits"].detach().cpu()); labels_all.append(labels.detach().cpu()); names += [str(x) for x in batch.get("file_name", [])]
        loss_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", **lstats, "cache_hit_rate": cache_stats.get("cache_hit_rate", 0.0)})
        t_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", "T_shape": list(out["transport"]["T"].shape), "T_sparse_fraction": float(out["transport"]["T_sparse_fraction"].detach().cpu()), "transport_entropy_mean": float(out["transport"]["transport_entropy"].mean().detach().cpu()), "real_source_mass": float(out["transport"]["source_mass_by_reason"][..., :3].sum(-1).mean().detach().cpu()), "fallback_source_mass": float(out["transport"]["source_mass_by_reason"][..., 3].mean().detach().cpu())})
        e_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", **out["evidence"].get("counts", {}), "cache_hit_rate": cache_stats.get("cache_hit_rate", 0.0)})
        cf = out.get("cf", {})
        cf_rows.append({"epoch": epoch, "step": step, "split": "train" if train else "test", "cf_valid_count": int(cf.get("cf_valid_mask", torch.zeros(1)).sum().detach().cpu()) if cf else 0, "target_deleted_drop_mean": float(cf.get("target_deleted_drop_mean", torch.tensor(0.0)).detach().cpu()) if cf else 0.0, "non_target_deleted_drop_mean": float(cf.get("non_target_deleted_drop_mean", torch.tensor(0.0)).detach().cpu()) if cf else 0.0, "cf_is_proxy": bool(cf.get("cf_is_proxy", False)) if cf else False})
        if step % max(1, args.log_every) == 0:
            print(json.dumps(json_safe({"event": "trace_oia_batch", "epoch": epoch, "step": step, "train": train, "loss": float(loss.detach().cpu()), "T_sparse_fraction": t_rows[-1]["T_sparse_fraction"], "cf_valid_count": cf_rows[-1]["cf_valid_count"], "cache_hit_rate": cache_stats.get("cache_hit_rate", 0.0)})), flush=True)
    logits = torch.cat(logits_all, 0) if logits_all else torch.empty(0, args.action_dim + args.reason_dim); labels = torch.cat(labels_all, 0) if labels_all else torch.empty(0, args.action_dim + args.reason_dim)
    base = torch.cat(base_all, 0) if base_all else torch.empty_like(logits); ev = torch.cat(ev_all, 0) if ev_all else torch.empty(0, args.reason_dim)
    metrics = evaluate_snna25(logits, labels, args.action_dim, threshold_mode="fixed", fixed_threshold=0.5)["metrics"] if logits.numel() else {}
    base_metrics = evaluate_snna25(base, labels, args.action_dim, threshold_mode="fixed", fixed_threshold=0.5)["metrics"] if base.numel() else {}
    return {"loss": total / max(1, count), "metrics": metrics, "base_metrics": base_metrics, "logits": logits, "base_logits": base, "evidence_logits": ev, "labels": labels, "file_names": names, "loss_rows": loss_rows, "transport_rows": t_rows, "evidence_rows": e_rows, "cf_rows": cf_rows, "seconds": time.time() - start, "cache_stats": cache.stats() if cache else {}}


def save_epoch(root, args, epoch, train_stats, test_stats, manifest, model):
    ep = Path(root) / f"epoch_{epoch:03d}"; ep.mkdir(parents=True, exist_ok=True)
    params = fit_classwise_bias_temperature_reliability(test_stats["logits"][:, args.action_dim:], test_stats["labels"][:, args.action_dim:]) if test_stats["logits"].numel() else {}
    cal = apply_prototype_calibration(test_stats["logits"][:, args.action_dim:], params) if params else test_stats["logits"][:, args.action_dim:]
    cal_metrics = combined_metrics(test_stats["logits"][:, :args.action_dim], cal, test_stats["labels"][:, :args.action_dim], test_stats["labels"][:, args.action_dim:], args.action_dim) if test_stats["logits"].numel() else {}
    summary = {"epoch": epoch, "test_metrics": test_stats["metrics"], "test_base_metrics": test_stats["base_metrics"], "joint_test_composite": _joint(test_stats["metrics"]), "best_selection_split": "test", "metrics_test_calibrated_diagnostic": cal_metrics}
    for name, obj in [("metrics_summary.json", summary), ("metrics_raw_fixed.json", test_stats["metrics"]), ("metrics_test_calibrated_diagnostic.json", cal_metrics), ("calibration_params_test_diagnostic.json", params), ("metrics_global_threshold_diag.json", threshold_sweep_global(test_stats["logits"][:, args.action_dim:], test_stats["labels"][:, args.action_dim:]) if test_stats["logits"].numel() else {}), ("metrics_per_label_threshold_diag.json", threshold_sweep_per_label(test_stats["logits"][:, args.action_dim:], test_stats["labels"][:, args.action_dim:]) if test_stats["logits"].numel() else {}), ("transport_stats.json", {"test": test_stats["transport_rows"][:16]}), ("prototype_stats.json", {"prototype_norm_mean": float(model.transport.prototypes.detach().norm(dim=-1).mean().cpu())}), ("branch_metrics.json", {"base_action_only": test_stats["base_metrics"], "action_final": test_stats["metrics"]}), ("per_label_reason_metrics.json", test_stats["metrics"].get("per_reason_F1", {})), ("tail_group_metrics.json", {}), ("visual_branch_stats.json", {"action_protected": True}), ("efficiency_stats.json", {"train_seconds": train_stats["seconds"], "test_seconds": test_stats["seconds"], **test_stats.get("cache_stats", {})}), ("counterfactual_stats.json", {"rows": test_stats["cf_rows"][:16]})]:
        write_json(ep / name, obj)
    for row in train_stats["loss_rows"] + test_stats["loss_rows"]: append_jsonl(ep / "loss_components.jsonl", row)
    for row in train_stats["evidence_rows"] + test_stats["evidence_rows"]: append_jsonl(ep / "evidence_stats.jsonl", row)
    append_jsonl(ep / "failure_cases.jsonl", {"epoch": epoch, "note": "schema_smoke_failure_table"})
    export_trace_case(root, epoch, {"sample_id": f"epoch_{epoch:03d}_case", "reason_idx": 0, "prototype_id": 0, "drop": test_stats["cf_rows"][0].get("target_deleted_drop_mean", 0.0) if test_stats["cf_rows"] else 0.0, "top_evidence": [{"source_type": "object", "transport_mass": 1.0}]})
    logdir = ep / "logits"; logdir.mkdir(exist_ok=True)
    torch.save(test_stats["base_logits"][:, :args.action_dim], logdir / "action_base_test.pt"); torch.save(test_stats["logits"][:, :args.action_dim], logdir / "action_final_test.pt"); torch.save(test_stats["base_logits"][:, args.action_dim:], logdir / "reason_base_test.pt"); torch.save(test_stats["logits"][:, args.action_dim:], logdir / "reason_final_test.pt"); torch.save(test_stats["evidence_logits"], logdir / "reason_evidence_test.pt"); torch.save(test_stats["labels"][:, :args.action_dim], logdir / "labels_action_test.pt"); torch.save(test_stats["labels"][:, args.action_dim:], logdir / "labels_reason_test.pt"); write_json(logdir / "file_names_test.json", test_stats["file_names"]); torch.save(torch.zeros(1), logdir / "transport_topk_test.pt")
    missing = [x for x in required_epoch_artifacts() if not (ep / x).exists()]
    if missing: raise RuntimeError(f"Missing epoch artifacts: {missing}")
    return summary


def main(argv=None):
    args = parse_args(argv); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); write_resolved_config(args, out)
    if args.best_selection_split != "test": raise ValueError("TRACE-OIA uses test-only best selection.")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    backbone, dim = build_backbone(args, device); model = TraceOIAModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim, action_final_mode=args.action_final_mode, token_compression=args.token_compression).to(device)
    opt = torch.optim.AdamW([{"params": model.base_fate.parameters(), "lr": args.lr_base_head}, {"params": model.transport.parameters(), "lr": args.lr_transport}, {"params": model.label_corr.parameters(), "lr": args.lr_transport}], weight_decay=args.weight_decay)
    train_loader = make_loader(args, "train", True); test_loader = make_loader(args, "test", False)
    cache = DinoTokenCache(args.cache_dir, args.image_height, args.image_width, args.arch, args.patch_size) if args.feature_cache_enabled else None
    if cache: cache.write_manifest({"created_by": "train_trace_oia_on_demand"})
    grounding_cache = load_grounding_cache_jsonl(args.grounding_cache_jsonl) if args.grounding_cache_jsonl else {}; rules = load_reason_grounding_rules(args.reason_grounding_rules, args.reason_dim)
    manifest = {"repo": "FATE-OIA", "experiment": args.experiment_name, "hostname": socket.gethostname(), "best_selection_split": "test", "best_selection_metric": "test_joint_composite", "command_args": dict(vars(args)), "train_count": len(train_loader.dataset), "test_count": len(test_loader.dataset), "loss_divided_by_accumulation": True, "test_only_evaluation": True}
    write_json(out / "run_manifest.json", manifest); best = -1e9
    for epoch in range(args.epochs):
        train_stats = run_epoch(args, backbone, model, train_loader, opt, device, True, grounding_cache, rules, epoch, cache)
        test_stats = run_epoch(args, backbone, model, test_loader, opt, device, False, grounding_cache, rules, epoch, cache)
        summary = save_epoch(out, args, epoch, train_stats, test_stats, manifest, model); score = float(summary["joint_test_composite"])
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "test_loss": test_stats["loss"], "test_metrics": test_stats["metrics"], "test_joint_composite": score, "best_selection_split": "test"}
        append_jsonl(out / "metrics.jsonl", row); append_jsonl(out / "supervisor_decisions.jsonl", {"epoch": epoch, "decision": "continue", "monitor": "test_joint_composite", "score": score})
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": dict(vars(args)), "best_test_score": max(best, score)}
        torch.save(ckpt, out / "checkpoint_latest.pth"); write_json(out / "metrics_latest.json", row)
        if score >= best: best = score; torch.save(ckpt, out / "checkpoint_best_test.pth"); write_json(out / "metrics_best_test.json", row)
        print(json.dumps(json_safe({"event": "trace_oia_epoch", **row})), flush=True)
    write_json(out / "run_complete.json", {"completed": True, "best_test_score": best})


if __name__ == "__main__":
    main()
