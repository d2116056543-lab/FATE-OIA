from pathlib import Path


def test_deployment_parameters_are_fit_before_test_is_applied():
    source = Path("fate_oia/engine/export_tida_deployment.py").read_text(encoding="utf-8")
    fit_body = source[source.index("def fit_deployment_parameters"):source.index("def apply_deployment")]
    assert 'test' not in fit_body.lower()
    assert '"train_calib", "train_audit", "test"' in source
    assert '"test_labels_used_for_parameter_fit": False' in source
