from fate_oia.datasets.aie_splits import stable_split_ids


def test_train_fit_excludes_calibration_and_audit_ids():
    ids = [f"frame_{index:04d}.jpg" for index in range(200)]
    split = stable_split_ids(ids, seed=17, calib_fraction=0.10, audit_count=20)

    fit = set(split["train_fit"])
    calib = set(split["train_calib"])
    audit = set(split["train_audit"])
    assert fit.isdisjoint(calib)
    assert fit.isdisjoint(audit)
    assert calib.isdisjoint(audit)
    assert fit | calib | audit == set(ids)
