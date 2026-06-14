import torch

from fate_oia.models.acpr_pair_memory import ACPRPairMemory, PairMiningThresholds
from fate_oia.losses.acpr_losses import matched_pair_logit_loss


def _batch():
    global_embed = torch.eye(4, 384)
    predicate = torch.ones(4, 32)
    action = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]).float()
    reason = torch.zeros(4, 21)
    reason[0, 5] = 1
    reason[2, 12] = 1
    contradiction = torch.ones(4, 21) * 0.8
    logits = torch.zeros(4, 21)
    logits[0, 5] = 0.0
    logits[1, 5] = 1.0
    logits[2, 12] = 0.0
    logits[3, 12] = 1.0
    reason_emb = torch.randn(4, 21, 384)
    return global_embed, predicate, action, reason, contradiction, logits, reason_emb


def test_hard_pair_mining_returns_active_pairs_and_loss():
    m = ACPRPairMemory()
    global_embed, predicate, action, reason, contradiction, logits, reason_emb = _batch()
    pairs = m.mine(
        ["a", "b", "c", "d"],
        global_embed,
        predicate,
        action,
        reason,
        contradiction,
        tail_indices=[12],
        reason_logits_current=logits,
        reason_embeddings_current=reason_emb,
        thresholds=PairMiningThresholds(action_sim_min=-1, visual_sim_min=-1, predicate_sim_min=-1, contradiction_min=0, tail_contradiction_min=0),
    )
    assert pairs["active_pair_count"] > 0
    assert pairs["hard_pair_count"] > 0
    loss = matched_pair_logit_loss(logits.clone().requires_grad_(True), pairs)
    assert float(loss.detach()) > 0


def test_tail_reason_multiplier_increases_weight():
    m = ACPRPairMemory()
    global_embed, predicate, action, reason, contradiction, logits, reason_emb = _batch()
    pairs = m.mine(
        ["a", "b", "c", "d"],
        global_embed,
        predicate,
        action,
        reason,
        contradiction,
        tail_indices=[12],
        reason_logits_current=logits,
        reason_embeddings_current=reason_emb,
        thresholds=PairMiningThresholds(action_sim_min=-1, visual_sim_min=-1, predicate_sim_min=-1, contradiction_min=0, tail_contradiction_min=0),
    )
    weights = pairs["pair_weights"]
    rids = pairs["pair_reason_ids"]
    assert weights[rids == 12].mean() > weights[rids == 5].mean()


def test_memory_negative_mining_uses_detached_logits_and_scan_cap():
    m = ACPRPairMemory(memory_size=64)
    global_embed, predicate, action, reason, contradiction, logits, reason_emb = _batch()
    memory_reason = torch.zeros_like(reason)
    memory_logits = torch.zeros_like(logits)
    memory_reason_emb = torch.randn_like(reason_emb)
    memory_logits[:, 5] = 1.25
    m.enqueue(
        ["m0", "m1", "m2", "m3"],
        global_embed,
        predicate,
        action,
        memory_reason,
        contradiction_scores=contradiction,
        reason_logits_detached=memory_logits,
        reason_embeddings_detached=memory_reason_emb,
    )
    pairs = m.mine(
        ["a", "b", "c", "d"],
        global_embed,
        predicate,
        action,
        reason,
        contradiction,
        tail_indices=[12],
        reason_logits_current=logits,
        reason_embeddings_current=reason_emb,
        max_memory_scan=2,
        thresholds=PairMiningThresholds(action_sim_min=-1, visual_sim_min=-1, predicate_sim_min=-1, contradiction_min=0, tail_contradiction_min=0),
    )
    assert pairs["pair_memory_count"] > 0
    assert pairs["hard_pair_count"] > 0
    assert pairs["pair_neg_is_memory"].any()
    assert pairs["pair_neg_logits_detached"].numel() == pairs["pair_count"]
    loss = matched_pair_logit_loss(logits.clone().requires_grad_(True), pairs)
    assert float(loss.detach()) > 0


def test_easy_pair_can_have_zero_active_loss():
    m = ACPRPairMemory()
    global_embed, predicate, action, reason, contradiction, _, reason_emb = _batch()
    reason[:, 12] = 0
    logits = torch.zeros(4, 21)
    logits[0, 5] = 2.0
    logits[1, 5] = 0.0
    pairs = m.mine(
        ["a", "b", "c", "d"],
        global_embed,
        predicate,
        action,
        reason,
        contradiction,
        tail_indices=[],
        reason_logits_current=logits,
        reason_embeddings_current=reason_emb,
        thresholds=PairMiningThresholds(action_sim_min=-1, visual_sim_min=-1, predicate_sim_min=-1, contradiction_min=0),
    )
    assert pairs["active_pair_count"] == 0
