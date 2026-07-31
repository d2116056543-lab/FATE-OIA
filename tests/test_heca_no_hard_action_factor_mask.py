from pathlib import Path

import torch
import yaml

from fate_oia.models.meter_schema import METERFactorSchema


def test_schema_uses_scalar_ownership_without_hard_compatibility() -> None:
    path = Path("configs/meter_factor_schema.yaml")
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))["factors"]
    assert all("compatible_actions" not in row for row in rows)
    schema = METERFactorSchema(path)
    ownership = torch.tensor(schema.action_ownership)
    assert ownership.shape == (21,)
    assert ownership[1] == 0.5
    assert ownership[14] == ownership[20] == 0
    assert torch.all(ownership[[i for i in range(21) if i not in (14, 20)]] > 0)

