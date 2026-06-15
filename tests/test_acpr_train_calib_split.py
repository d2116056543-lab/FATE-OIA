from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


class _Dataset:
    def __init__(self, n=100):
        self.items = [{"file_name": f"sample_{i:04d}.jpg"} for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_make_train_calib_indices_is_deterministic_and_disjoint():
    ds = _Dataset(100)
    main_a, calib_a = make_train_calib_indices(ds, calib_fraction=0.1, seed=20260615)
    main_b, calib_b = make_train_calib_indices(ds, calib_fraction=0.1, seed=20260615)

    assert main_a == main_b
    assert calib_a == calib_b
    assert len(calib_a) == 10
    assert set(main_a).isdisjoint(calib_a)
    assert sorted(main_a + calib_a) == list(range(100))
