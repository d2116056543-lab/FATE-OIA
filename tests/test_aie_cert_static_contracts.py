from pathlib import Path


def test_formal_graph_does_not_import_old_aie_components():
    files=list(Path('fate_oia').rglob('aie_cert_*.py'))
    forbidden=('aie_evidence_interface','aie_contribution_head','aie_reason_rereader','aie_predicate_naming','utils.aie_counterfactual','models.aie_oia_model')
    violations=[(str(p),x) for p in files for x in forbidden if x in p.read_text(encoding='utf-8')]
    assert violations==[]
