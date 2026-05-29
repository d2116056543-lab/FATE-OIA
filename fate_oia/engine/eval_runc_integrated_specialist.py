from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.train_runc_integrated_specialist import (
    RUNC_ART,
    build_base_model,
    make_args,
    run_epoch,
    write_json,
)
from fate_oia.models.action_set_head import build_action_patterns
from fate_oia.models.runc_integrated_specialist import RunCIntegratedSpecialist
import fate_oia.engine.train_fate_oia as t


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a RunC integrated specialist checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--runc_args", default=str(RUNC_ART / "args.json"))
    ap.add_argument("--runc_checkpoint", default=str(RUNC_ART / "checkpoint_best_test.pth"))
    cli = ap.parse_args()
    args = make_args(Namespace(
        runc_args=cli.runc_args,
        runc_checkpoint=cli.runc_checkpoint,
        output_dir=str(Path(cli.output).parent),
        data_root=r"E:\sbw\FATE_Drive\fate_oia_worktree\dataset\BDD-OIA",
        raw_root=r"E:\sbw\FATE_Drive\fate_oia_worktree\raw_data\BDD-OIA",
        pretrained_weights=r"E:\sbw\FATE_Drive\fate_oia_worktree\ckp\reference\dino_deitsmall8_pretrain.pth",
        device=cli.device,
        max_test_samples=cli.max_test_samples,
        max_train_samples=0,
        max_val_samples=0,
        batch_size=4,
        gradient_accumulation_steps=8,
        epochs=24,
        lr_base=1e-5,
        lr_specialist=1e-4,
        lr_bias=5e-3,
        weight_decay=1e-4,
        min_lr=1e-6,
        scheduler='cosine',
        freeze_base=False,
        freeze_dino=True,
        reason_specialist=True,
        action_set_head=True,
        evidence_aux=False,
        loss_reason_asl=1.0,
        loss_reason_ranking=0.15,
        loss_sigmoid_f1=0.05,
        loss_action_pattern=0.10,
        loss_action_preserve=0.02,
        loss_non_tail_distill=0.02,
        loss_delta_l2=0.01,
        loss_evidence_distill=0.05,
        loss_final_action=0.25,
        ranking_margin=0.5,
        ranking_hard_k=5,
        num_workers=0,
        log_every=1000000000,
        max_saved_token_stats=0,
    ))
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    backbone, dim = t.build_backbone(args, device)
    base = build_base_model(args, dim, device)
    dummy = torch.optim.AdamW(base.parameters(), lr=args.lr_base)
    t.load_resume_checkpoint(args.resume, base, dummy, device=device, resume_optimizer=False, strict=True)
    ckpt = torch.load(cli.checkpoint, map_location=device)
    pattern_matrix = ckpt.get('pattern_matrix')
    model = RunCIntegratedSpecialist(base, dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim, pattern_matrix=pattern_matrix.to(device) if pattern_matrix is not None else None).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_specialist)
    loader = t.make_loader(args, 'test', False)
    stats = run_epoch(args, backbone, model, loader, optimizer, device, model.action_set_head.pattern_matrix.detach().cpu(), train=False, epoch=int(ckpt.get('epoch', -1)))
    result = {"checkpoint": cli.checkpoint, "count": stats['count'], "metrics": stats['metrics'], "base_metrics": stats['base_metrics'], "branch_metrics": stats['branch_metrics']}
    write_json(cli.output, result)
    print(json.dumps(t._json_safe(result), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
