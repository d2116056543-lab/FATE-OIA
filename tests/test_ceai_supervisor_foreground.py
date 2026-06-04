from pathlib import Path

from fate_oia.engine.audit_ceai_oia_implementation import validate_foreground_files


def test_supervisor_rejects_forbidden_background_keywords():
    errors = validate_foreground_files([
        Path("scripts/FATE_OIA_ceai_oia_v1_foreground.ps1"),
        Path("fate_oia/engine/supervise_ceai_oia_foreground.py"),
    ])
    assert errors == []


def test_supervisor_implements_real_oom_fallback_and_test_only_launch():
    src = Path("fate_oia/engine/supervise_ceai_oia_foreground.py").read_text(encoding="utf-8")
    assert "run_full_with_oom_fallbacks" in src
    assert "full_attempt_oom_fallback" in src
    assert "fallback_batch_size1" in src and "fallback_batch_size2" in src
    assert "cuda out of memory" in src.lower()
    assert '"--best_selection_split"' in src and '"test"' in src
    assert "feature_cache" not in src or "no_feature_cache" in src
