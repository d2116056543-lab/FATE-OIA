import inspect

from fate_oia.utils import tida_temporal_interventions


def test_interventions_operate_only_on_query_history():
    source = inspect.getsource(tida_temporal_interventions)
    assert "dino" not in source.lower()
    assert "target_image" not in source
