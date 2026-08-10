from fate_oia.utils.vetra_artifacts import REQUIRED_PROBE_ARTIFACTS, validate_probe_artifacts, write_json


def test_probe_artifact_validator_reports_only_missing_files(tmp_path):
    for name in REQUIRED_PROBE_ARTIFACTS[:-1]:
        path = tmp_path / name
        if name.endswith(".jsonl"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            write_json(path, {})
    assert validate_probe_artifacts(tmp_path) == [REQUIRED_PROBE_ARTIFACTS[-1]]

