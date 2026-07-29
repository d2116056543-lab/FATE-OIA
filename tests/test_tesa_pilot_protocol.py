from fate_oia.engine.evaluate_tesa_pilot import validate_pilot_protocol


def test_pilot_protocol_rejects_tiny_mock_or_incomplete_runs() -> None:
    expected = {
        "train_main": 4096,
        "train_audit": 1024,
        "train_calib": 512,
        "test": 512,
        "epochs": 4,
        "seed": 20260729,
    }
    manifest = {
        "seed": 20260729,
        "use_mock_dino": False,
        "runtime_subset_counts": {
            "train_main": 4096,
            "train_audit": 1024,
            "train_calib": 512,
            "test": 512,
        },
    }
    assert validate_pilot_protocol(manifest, expected, completed_epochs=4) == []
    manifest["use_mock_dino"] = True
    assert "mock_dino" in validate_pilot_protocol(manifest, expected, completed_epochs=4)
    manifest["use_mock_dino"] = False
    manifest["runtime_subset_counts"]["train_main"] = 16
    failures = validate_pilot_protocol(manifest, expected, completed_epochs=3)
    assert "train_main" in failures
    assert "epochs" in failures
