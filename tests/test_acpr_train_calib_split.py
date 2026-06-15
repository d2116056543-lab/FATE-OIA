from __future__ import annotations

from dataclasses import dataclass

from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


@dataclass
class _Sample:
    file_name: str


class _DatasetWithSamples:
    def __init__(self) -> None:
        self.samples = [_Sample(f"sample_{idx:03d}.jpg") for idx in range(10)]
        self.getitem_calls = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        self.getitem_calls += 1
        raise AssertionError("make_train_calib_indices should not load image items when samples metadata exists")


def test_make_train_calib_indices_uses_samples_without_getitem() -> None:
    dataset = _DatasetWithSamples()
    main, calib = make_train_calib_indices(dataset, calib_fraction=0.2, seed=123)

    assert len(calib) == 2
    assert len(main) == 8
    assert sorted(main + calib) == list(range(10))
    assert dataset.getitem_calls == 0
