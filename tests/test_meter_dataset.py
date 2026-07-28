from pathlib import Path

from fate_oia.datasets.meter_dataset import fixed_meter_split_indices
from fate_oia.datasets.meter_grounding_index import METERGroundingIndex


def test_meter_fixed_splits_are_disjoint_and_reproducible() -> None:
    names = [f"frame_{index:04d}.jpg" for index in range(100)]
    one = fixed_meter_split_indices(names, audit_fraction=0.08, calib_fraction=0.10, seed=9)
    two = fixed_meter_split_indices(names, audit_fraction=0.08, calib_fraction=0.10, seed=9)
    assert one == two
    assert not (set(one["main"]) & set(one["audit"]))
    assert not (set(one["main"]) & set(one["calib"]))
    assert not (set(one["audit"]) & set(one["calib"]))
    assert sum(len(indices) for indices in one.values()) == len(names)


def test_meter_index_uses_lru_metadata_and_does_not_expose_geometry_to_test() -> None:
    index = METERGroundingIndex(Path("unused"), schema_path=Path("configs/meter_factor_schema.yaml"))
    index._records["sample"] = {"source_complete": False, "objects": []}
    train = index.signed_target("sample.jpg", split="train")
    test = index.signed_target("sample.jpg", split="test")
    assert train is not None
    assert test is None
