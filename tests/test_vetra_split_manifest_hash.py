from fate_oia.datasets.aie_splits import ids_sha256, stable_split_ids


def test_vetra_split_is_deterministic_disjoint_and_hashable():
    ids=[str(i) for i in range(100)]
    a=stable_split_ids(ids,20260810,.1,20); b=stable_split_ids(ids,20260810,.1,20)
    assert a==b and ids_sha256(a["train_main"])==ids_sha256(b["train_main"])
    assert not set(a["train_main"]) & set(a["train_calib"])
    assert not set(a["train_main"]) & set(a["train_audit"])
