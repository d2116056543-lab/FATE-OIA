from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.mosaic_native_semantics import load_icdor_ontology
from fate_oia.models.mosaic_icdor_dual_reason_decoder import MOSAICICDORLatentReasonDecoder
from fate_oia.models.mosaic_target_sparse_router import MOSAICTargetSparseRouter


def test_disallowed_huge_semantics_cannot_steal_entmax_mass() -> None:
    torch.manual_seed(4)
    ontology = load_icdor_ontology(Path("configs"))
    router = MOSAICTargetSparseRouter(ontology, dim=16)
    router.set_certificate_tiers(["certified"] * router.factor_count)
    factor = torch.randn(1, router.factor_count, 16)
    action = torch.randn(1, 4, 16)
    positive = torch.rand(1, router.factor_count)
    negative = 1.0 - positive
    base = router(factor, positive, negative, action, route_mode="shadow")
    allowed_any = base["active_edge_mask"].any(dim=(0, 2))
    disallowed = torch.nonzero(~allowed_any, as_tuple=False).flatten()
    assert disallowed.numel() > 0
    attacked = factor.clone()
    attacked[:, disallowed] = 1e6
    changed = router(attacked, positive, negative, action, route_mode="shadow")
    for key in ("support_weights", "veto_weights", "support_dustbin", "veto_dustbin"):
        assert (base[key] - changed[key]).abs().max().item() <= 1e-6
    for direction, key in enumerate(("support_weights", "veto_weights")):
        mask = ~base["active_edge_mask"][direction]
        assert torch.count_nonzero(changed[key].masked_select(mask.unsqueeze(0))) == 0


def test_disallowed_reason_factor_cannot_change_escape_allowed_weights_or_logits() -> None:
    torch.manual_seed(5)
    ontology = load_icdor_ontology(Path("configs"))
    decoder = MOSAICICDORLatentReasonDecoder(
        ontology, dim=16, decoder_layers=1, self_attention_heads=4, highres_topk=8, midres_topk=4
    )
    factor_count = len(ontology["factors"])
    pyramid = {
        "F_hi": torch.randn(1, 16, 45, 80),
        "F_mid": torch.randn(1, 16, 23, 40),
        "F_ctx": torch.randn(1, 16, 12, 20),
    }
    factor = torch.randn(1, factor_count, 16)
    masks = torch.rand(1, factor_count, 45, 80)
    enabled = torch.ones(factor_count, dtype=torch.bool)
    base = decoder(pyramid, factor, masks, enabled)
    disallowed = ~decoder.reason_factor_allow_mask[0]
    assert disallowed.any()
    attacked = factor.clone()
    attacked[:, disallowed] = 1e6
    changed = decoder(pyramid, attacked, masks, enabled)
    assert (base["reason_factor_router_weights"][:, 0] - changed["reason_factor_router_weights"][:, 0]).abs().max().item() <= 1e-6
    assert (base["reason_escape_weight"][:, 0] - changed["reason_escape_weight"][:, 0]).abs().max().item() <= 1e-6
    assert (base["reason_logits_latent"][:, 0] - changed["reason_logits_latent"][:, 0]).abs().max().item() <= 1e-6
    assert torch.count_nonzero(changed["reason_factor_router_weights"][:, 0, disallowed]) == 0
