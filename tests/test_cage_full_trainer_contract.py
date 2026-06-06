from __future__ import annotations

import argparse
import importlib

import torch


def test_cage_parser_accepts_direct_image_training_args():
    mod = importlib.import_module("fate_oia.engine.train_cage_oia")
    args = mod.build_parser().parse_args([
        "--output_dir", ".background_runs/cage_contract",
        "--data_root", "E:/sbw/BDD-OIA/data",
        "--raw_root", "E:/sbw/BDD-OIA",
        "--max_train_samples", "2",
        "--max_test_samples", "2",
        "--epochs", "1",
        "--batch_size", "1",
    ])
    assert args.smoke_only is False
    assert args.max_test_samples == 2
    assert args.threshold_mode == "fixed"


def test_selected_vs_random_drop_uses_real_masked_forwards():
    mod = importlib.import_module("fate_oia.engine.train_cage_oia")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3, 4)
            with torch.no_grad():
                self.lin.weight.fill_(0.1)
                self.lin.bias.zero_()

        def forward(self, tokens):
            pooled = tokens.mean(1)
            logits = self.lin(pooled)
            return {"action_logits": logits, "reason_logits": torch.zeros(tokens.shape[0], 21, device=tokens.device)}

    model = Tiny()
    tokens = torch.randn(2, 6, 3)
    labels = torch.ones(2, 25)
    pred = {
        "action_logits": model(tokens)["action_logits"],
        "evidence": {"topk_indices": torch.tensor([[[0, 1], [2, 3], [1, 4], [0, 5]], [[0, 2], [1, 3], [2, 4], [1, 5]]])},
    }
    rows = mod.selected_vs_random_action_drop(model, tokens, labels, pred, action_dim=4, max_labels=4)
    assert len(rows) == 4
    assert all(r["available"] is True for r in rows)
    assert all(r["judge"] == "action_gt_loss_drop" for r in rows)
