from pathlib import Path

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset


def test_bdd_oia_split_counts_and_dims():
    if not Path("dataset/BDD-OIA").exists():
        return
    expected = {"train": 16082, "val": 2270, "test": 4572}
    for split, count in expected.items():
        ds = BDDOIAMultiTaskDataset("dataset/BDD-OIA", "raw_data/BDD-OIA", split=split, action_dim=4)
        assert len(ds) == count
        item = ds[0]
        assert item["action"].shape[0] == 4
        assert item["reason"].shape[0] == 21
        assert Path(item["image_path"]).exists()


def test_bdd_oia_action_dim5_if_available():
    if not Path("dataset/BDD-OIA").exists():
        return
    ds = BDDOIAMultiTaskDataset("dataset/BDD-OIA", "raw_data/BDD-OIA", split="train", action_dim=5)
    assert ds[0]["action"].shape[0] == 5
