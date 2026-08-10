from fate_oia.datasets.aie_splits import stable_split_ids


def test_dice_train_main_calib_audit_are_disjoint_and_complete():
    ids = [f"sample-{i}" for i in range(100)]
    split = stable_split_ids(ids, seed=7, calib_fraction=0.1, audit_count=20)
    main, calib, audit = map(set, (split["train_main"], split["train_calib"], split["train_audit"]))
    assert not main & calib
    assert not main & audit
    assert not calib & audit
    assert main | calib | audit == set(ids)
