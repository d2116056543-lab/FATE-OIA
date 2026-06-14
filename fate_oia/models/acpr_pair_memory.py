from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ACPRPairMemory(nn.Module):
    def __init__(self, dim: int = 384, memory_size: int = 8192, tail_multiplier: float = 2.0) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.tail_multiplier = tail_multiplier
        self.proj = nn.Linear(dim, dim)

    def forward(self, label_nodes: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(label_nodes.mean(1)), dim=-1)

    def mine_pairs(
        self,
        embeddings: torch.Tensor,
        action: torch.Tensor,
        reason: torch.Tensor,
        tail_indices: list[int] | None = None,
        global_embedding: torch.Tensor | None = None,
        predicate_probs: torch.Tensor | None = None,
        contradiction_scores: torch.Tensor | None = None,
        file_names: list[str] | None = None,
        max_pairs: int | None = None,
    ) -> dict[str, torch.Tensor | int | list[str]]:
        b = int(reason.shape[0])
        device = reason.device
        max_pairs = int(max_pairs or min(self.memory_size, 4096))
        if b < 2:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {"pair_pos_indices": empty, "pair_neg_indices": empty, "pair_reason_ids": empty, "pair_weights": torch.empty(0, device=device), "positive_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "contrast_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "pair_count": 0, "hard_negative_count": 0, "tail_pair_count": 0}
        emb = F.normalize(global_embedding if global_embedding is not None else embeddings, dim=-1)
        pred = F.normalize(predicate_probs if predicate_probs is not None else torch.zeros(b, 1, device=device), dim=-1)
        action_sim = F.cosine_similarity(action[:, None, :], action[None, :, :], dim=-1).clamp(-1, 1)
        visual_sim = emb @ emb.t()
        predicate_sim = pred @ pred.t()
        tail = set(int(x) for x in (tail_indices or []))
        pos_out: list[int] = []
        neg_out: list[int] = []
        rid_out: list[int] = []
        wt_out: list[float] = []
        hard_count = 0
        for r in range(reason.shape[1]):
            pos_idx = torch.where(reason[:, r] > 0.5)[0]
            neg_idx = torch.where(reason[:, r] <= 0.5)[0]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            for pi in pos_idx.tolist():
                score = 0.40 * action_sim[pi, neg_idx] + 0.30 * visual_sim[pi, neg_idx] + 0.20 * predicate_sim[pi, neg_idx]
                if contradiction_scores is not None:
                    score = score + 0.30 * contradiction_scores[neg_idx, r]
                top = neg_idx[torch.argsort(score, descending=True)[:2]]
                for ni in top.tolist():
                    pos_out.append(pi); neg_out.append(ni); rid_out.append(r)
                    c = float(contradiction_scores[ni, r].detach().cpu()) if contradiction_scores is not None else 0.5
                    if c > 0.5:
                        hard_count += 1
                    wt_out.append((2.0 if r in tail else 1.0) * (0.5 + c))
                    if len(pos_out) >= max_pairs:
                        break
                if len(pos_out) >= max_pairs:
                    break
            if len(pos_out) >= max_pairs:
                break
        if not pos_out:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {"pair_pos_indices": empty, "pair_neg_indices": empty, "pair_reason_ids": empty, "pair_weights": torch.empty(0, device=device), "positive_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "contrast_pairs": torch.empty(0, 2, dtype=torch.long, device=device), "pair_count": 0, "hard_negative_count": 0, "tail_pair_count": 0}
        pair_reason_ids = torch.tensor(rid_out, dtype=torch.long, device=device)
        pos_t = torch.tensor(pos_out, dtype=torch.long, device=device)
        neg_t = torch.tensor(neg_out, dtype=torch.long, device=device)
        legacy_pairs = torch.stack([pos_t, neg_t], dim=1)
        return {
            "pair_pos_indices": pos_t,
            "pair_neg_indices": neg_t,
            "pair_reason_ids": pair_reason_ids,
            "pair_weights": torch.tensor(wt_out, dtype=torch.float32, device=device),
            "positive_pairs": legacy_pairs,
            "contrast_pairs": legacy_pairs,
            "pair_action_sim": action_sim[pos_t, neg_t].detach(),
            "pair_visual_sim": visual_sim[pos_t, neg_t].detach(),
            "pair_predicate_sim": predicate_sim[pos_t, neg_t].detach(),
            "pair_contradiction": contradiction_scores[neg_t, pair_reason_ids].detach() if contradiction_scores is not None else torch.zeros(len(pos_out), device=device),
            "pair_count": len(pos_out),
            "hard_negative_count": hard_count,
            "tail_pair_count": int(sum(1 for r in rid_out if r in tail)),
            "file_names": file_names or [],
        }
