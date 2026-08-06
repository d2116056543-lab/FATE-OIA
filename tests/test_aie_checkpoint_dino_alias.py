from collections import OrderedDict

import pytest
import torch

from fate_oia.engine.train_aie_oia import canonical_model_state_dict


def test_checkpoint_canonicalization_removes_only_verified_dino_vproj_alias():
    state = OrderedDict(
        {
            "foundation.dino.backbone.blocks.0.attn.proj.weight": torch.ones(2, 2),
            "foundation.dino.backbone.blocks.0.attn.vproj.weight": torch.ones(2, 2),
            "head.weight": torch.ones(1),
        }
    )
    clean = canonical_model_state_dict(state)
    assert "foundation.dino.backbone.blocks.0.attn.vproj.weight" not in clean
    assert "foundation.dino.backbone.blocks.0.attn.proj.weight" in clean
    assert "head.weight" in clean


def test_checkpoint_canonicalization_rejects_unmatched_alias():
    with pytest.raises(RuntimeError, match="unmatched DINO vproj alias"):
        canonical_model_state_dict({"foundation.dino.backbone.blocks.0.attn.vproj.weight": torch.ones(1)})
