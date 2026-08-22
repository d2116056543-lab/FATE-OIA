from pathlib import Path

from fate_oia.engine.train_tida_oia import completion_pass
from fate_oia.utils.tida_contracts import resolve_schedule_total_updates, schedule_values


def test_update_cap_is_available_only_to_nonformal_runs():
    source = Path("fate_oia/engine/train_tida_oia.py").read_text(encoding="utf-8")
    assert 'formal full training forbids max_optimizer_updates' in source
    assert 'optimizer_update >= int(max_optimizer_updates)' in source


def test_smoke_completion_fails_closed_before_requested_updates():
    assert not completion_pass("smoke", completed_epochs=1, optimizer_updates=7, max_optimizer_updates=50)
    assert completion_pass("smoke", completed_epochs=10, optimizer_updates=50, max_optimizer_updates=50)
    assert completion_pass("smoke", completed_epochs=1, optimizer_updates=7, max_optimizer_updates=None)
    assert not completion_pass("full", completed_epochs=9, optimizer_updates=500, max_optimizer_updates=None)
    assert completion_pass("full", completed_epochs=10, optimizer_updates=500, max_optimizer_updates=None)


def test_update_cap_does_not_silently_compress_the_formal_temporal_schedule():
    total = resolve_schedule_total_updates(updates_per_epoch=104, configured_epochs=10)
    assert total == 1040
    assert schedule_values(50, total)["temporal_scale"] == 0.0
    compressed = resolve_schedule_total_updates(
        updates_per_epoch=104, configured_epochs=10, schedule_total_updates=50
    )
    assert compressed == 50 and schedule_values(10, compressed)["temporal_scale"] == 1.0
