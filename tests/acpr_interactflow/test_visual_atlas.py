from __future__ import annotations

import json

import torch

from fate_oia.acpr_interactflow.artifacts import write_json
from fate_oia.engine.export_acpr_interactflow_visuals import main as export_main
from fate_oia.explain.acpr_interactflow_atlas import build_atlas


def test_visual_export_and_atlas_use_tensor_artifacts(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    final = torch.tensor([[0.2, 0.1, -0.1]])
    global_logits = torch.tensor([[0.1, 0.0, -0.2]])
    flow = torch.tensor([[0.05, 0.05, 0.05]])
    calib = final - global_logits - flow
    torch.save(final, run / "logits_action_test.pt")
    torch.save(torch.zeros(1, 29), run / "logits_exp29_test.pt")
    torch.save(torch.tensor([0]), run / "labels_action_test.pt")
    torch.save(torch.zeros(1, 29), run / "labels_exp29_test.pt")
    torch.save(global_logits, run / "logits_action_global_test.pt")
    torch.save(flow, run / "logits_action_flow_delta_test.pt")
    torch.save(calib, run / "logits_action_calibration_delta_test.pt")
    torch.save(torch.zeros(1, 16, 3), run / "ledger_gated_state_contributions_test.pt")
    write_json(run / "file_names_test.json", ["sample_0001"])
    write_json(run / "metrics_latest.json", {"joint": 0.1})

    out = tmp_path / "visual"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export",
            "--metrics",
            str(run / "metrics_latest.json"),
            "--output_dir",
            str(out),
            "--max_cases",
            "1",
        ],
    )
    export_main()
    assert (out / "case_0000" / "decision_ledger.json").exists()
    assert (out / "case_0000" / "decision_waterfall.png").exists()
    assert "Dynamic Interaction Decision Ledger" in (out / "case_0000" / "report.html").read_text(encoding="utf-8")

    intervention = tmp_path / "intervention_audit.json"
    write_json(
        intervention,
        {
            "results": {
                "evidence_tube_off": {"action_prob_l1_delta": 0.2, "exp29_prob_l1_delta": 0.1},
                "equal_mass_random": {"action_prob_l1_delta": 0.02, "exp29_prob_l1_delta": 0.01},
            },
            "nonzero_delta_count": 2,
        },
    )
    manifest = build_atlas([out / "case_0000"], out / "atlas.html", run / "metrics_latest.json", intervention)
    assert manifest["case_count"] == 1
    assert manifest["faithfulness"]["evidence_specificity_proxy"] > 0
    assert "Dataset Atlas" in (out / "atlas.html").read_text(encoding="utf-8")

