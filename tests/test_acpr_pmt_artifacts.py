from pathlib import Path
from fate_oia.utils.acpr_pmt_artifacts import write_pmt_epoch_artifacts


def test_pmt_artifacts_are_non_placeholder(tmp_path: Path):
    write_pmt_epoch_artifacts(tmp_path, 0, {
        "predicate_patch_alignment": {"mass_mean": 0.5, "available": True},
        "triadic_mediator_stats": {"delta_abs_mean": 0.0, "available": True},
        "pmt_phase_schedule": {"phase": "warmup", "available": True},
    })
    assert (tmp_path / "predicate_patch_alignment.jsonl").read_text().strip()
    assert "mass_mean" in (tmp_path / "predicate_patch_alignment.jsonl").read_text()
