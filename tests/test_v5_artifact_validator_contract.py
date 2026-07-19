from __future__ import annotations

import inspect

from fate_oia.utils import mosaic_icdor_artifacts as artifacts


def test_artifact_validator_requires_v5_epoch_zero_target_utility() -> None:
    """V5 artifact validation must not preserve the V4 FOUNDATION abstention."""
    source = inspect.getsource(artifacts.validate_icdor_artifact_schema)

    assert '"v5_credo_map"' in source
    assert '"audit_level"' in source
    assert '"JOINT_SHADOW"' in source
    assert '"ADMISSION_CONSOLIDATION"' in source
    assert '"FOUNDATION"' not in source
    assert '"v4_credo"' not in source
