from pathlib import Path

import yaml


def test_latent_factors_are_not_grounded_or_action_owned() -> None:
    rows = yaml.safe_load(Path("configs/meter_factor_schema.yaml").read_text())["factors"]
    for index in (14, 20):
        assert rows[index]["groundability"] == "latent"
        assert rows[index]["action_owned"] == 0
