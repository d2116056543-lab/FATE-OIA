import json
from pathlib import Path

import torch

from fate_oia.engine.offline_score_fusion_sweep import run_phase_b_fusion_sweep


def _write_branch(root: Path, epoch: int, *, names, order, reason_good: bool) -> Path:
    epoch_dir = root / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True)
    action_labels = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    reason_labels = torch.zeros(4, 21)
    reason_labels[0, 0] = 1
    reason_labels[1, 0] = 1
    reason_labels[2, 1] = 1
    reason_labels[3, 1] = 1
    action_logits = torch.where(action_labels > 0, torch.full_like(action_labels, 4.0), torch.full_like(action_labels, -4.0))
    if reason_good:
        reason_logits = torch.where(reason_labels > 0, torch.full_like(reason_labels, 4.0), torch.full_like(reason_labels, -4.0))
    else:
        reason_logits = torch.zeros_like(reason_labels)
    idx = torch.tensor(order, dtype=torch.long)
    torch.save(action_logits[idx], epoch_dir / "logits_action_test.pt")
    torch.save(action_logits[idx], epoch_dir / "logits_action_fused_test.pt")
    torch.save(reason_logits[idx], epoch_dir / "logits_reason_test.pt")
    torch.save(action_labels[idx], epoch_dir / "labels_action_test.pt")
    torch.save(reason_labels[idx], epoch_dir / "labels_reason_test.pt")
    (epoch_dir / "file_names_test.json").write_text(json.dumps([names[i] for i in order]), encoding="utf-8")
    with (root / "metrics_summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": epoch, "test_joint": 0.1 + epoch}) + "\n")
    return epoch_dir


def test_phase_b_sweep_aligns_file_names_and_writes_plan_outputs(tmp_path):
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    run_c = tmp_path / "run_c"
    s1 = tmp_path / "s1"
    output = tmp_path / "phase_b"
    run_c.mkdir()
    s1.mkdir()
    _write_branch(run_c, 14, names=names, order=[0, 1, 2, 3], reason_good=False)
    _write_branch(s1, 19, names=names, order=[2, 0, 3, 1], reason_good=True)

    payload = run_phase_b_fusion_sweep(
        run_c_dir=run_c,
        s1_dir=s1,
        output_dir=output,
        split="test",
        action_dim=4,
        reason_dim=21,
    )

    assert (output / "fusion_sweep_test.json").exists()
    assert (output / "threshold_sweep_test.json").exists()
    assert (output / "phase_b_decision.json").exists()
    assert payload["alignment"]["s1_reordered_to_run_c"] is True
    assert payload["decision"]["logit_fusion_promising"] is True
    assert payload["best_fusion"]["Exp_mF1"] > payload["run_c_fixed"]["Exp_mF1"]


def test_phase_b_prefers_root_best_test_artifacts_over_metrics_epoch(tmp_path):
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    run_c = tmp_path / "run_c"
    s1 = tmp_path / "s1"
    output = tmp_path / "phase_b"
    run_c.mkdir()
    s1.mkdir()
    _write_branch(run_c, 16, names=names, order=[0, 1, 2, 3], reason_good=False)
    best_epoch = _write_branch(run_c, 14, names=names, order=[0, 1, 2, 3], reason_good=True)
    for source_name, best_name in [
        ("logits_action_fused_test.pt", "logits_action_fused_best_test.pt"),
        ("logits_reason_test.pt", "logits_reason_best_test.pt"),
        ("labels_action_test.pt", "labels_action_best_test.pt"),
        ("labels_reason_test.pt", "labels_reason_best_test.pt"),
    ]:
        (run_c / best_name).write_bytes((best_epoch / source_name).read_bytes())
    (run_c / "file_names_best_test.json").write_text((best_epoch / "file_names_test.json").read_text(encoding="utf-8"), encoding="utf-8")
    _write_branch(s1, 19, names=names, order=[0, 1, 2, 3], reason_good=False)

    payload = run_phase_b_fusion_sweep(
        run_c_dir=run_c,
        s1_dir=s1,
        output_dir=output,
        split="test",
        action_dim=4,
        reason_dim=21,
    )

    assert payload["inputs"]["run_c"]["epoch_dir"] == str(run_c)
    assert payload["run_c_fixed"]["Exp_mF1"] > payload["score_v2_fixed"]["Exp_mF1"]
