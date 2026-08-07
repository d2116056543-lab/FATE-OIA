from fate_oia.utils.aie_cert_artifacts import REQUIRED_EPOCH,validate_epoch


def test_missing_artifacts_are_reported(tmp_path):
    assert set(validate_epoch(tmp_path))==set(REQUIRED_EPOCH)
    for name in REQUIRED_EPOCH: (tmp_path/name).write_text('{}')
    assert validate_epoch(tmp_path)==[]
