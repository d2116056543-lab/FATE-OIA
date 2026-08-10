def test_vetra_runtime_entrypoints_import():
    from fate_oia.engine import train_vetra_oia_probe, vetra_common

    assert callable(train_vetra_oia_probe.main)
    assert callable(vetra_common.build_vetra_model)
