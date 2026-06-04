import inspect

from fate_oia.engine.audit_ceai_oia_implementation import run_static_audit
from fate_oia.models.ceai_oia_model import CEAIOIAFeatureModel
from fate_oia.models.ceai_pair_sparse_attention import TaskGuidedPairSparseAttention
from fate_oia.models.ceai_router import ParetoSafeRouter


def test_audit_static_gates_and_source_markers():
    errors = run_static_audit(require_smoke_artifacts=False)
    assert errors == []
    assert "base_action_logits" in inspect.getsource(CEAIOIAFeatureModel.forward)
    assert "topk" in inspect.getsource(TaskGuidedPairSparseAttention.forward)
    assert "base_action_logits + action_delta" in inspect.getsource(ParetoSafeRouter.forward)
