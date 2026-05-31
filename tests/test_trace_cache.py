import torch
from fate_oia.datasets.dino_token_cache import DinoTokenCache

def test_cache_build_and_read(tmp_path):
    c = DinoTokenCache(tmp_path); c.put("a.jpg", torch.randn(4, 8), torch.ones(3))
    assert c.get("a.jpg")["file_name"] == "a.jpg"
    assert c.audit(["a.jpg"])["cache_hit_rate"] == 1.0
