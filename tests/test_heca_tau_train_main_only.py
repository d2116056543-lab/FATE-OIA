import torch

from fate_oia.engine.prepare_heca_static_artifacts import build_tau_from_train_main


def test_tau_reads_train_main_only() -> None:
    calls: list[str] = []

    def provider(name: str) -> dict[str, torch.Tensor]:
        calls.append(name)
        return {
            "factor_observability": torch.tensor([1.0, 0.0]),
            "factor_observability_valid": torch.tensor([1.0, 1.0]),
        }

    tau, metadata = build_tau_from_train_main(
        ["main-a", "audit", "main-b", "calib"], [0, 2], provider, ["actor", "actor"]
    )
    assert calls == ["main-a", "main-b"]
    assert metadata["fit_split"] == "train_main"
    assert tau.shape == (2,)
