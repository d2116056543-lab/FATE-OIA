from __future__ import annotations

import torch


class EvidenceMemoryBank:
    def __init__(self, capacity: int = 2048, dim: int = 384) -> None:
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.tokens: list[torch.Tensor] = []
        self.reason_ids: list[int] = []

    def enqueue(self, evidence_tokens: torch.Tensor, reason_evidence_mask: torch.Tensor) -> None:
        with torch.no_grad():
            for b in range(evidence_tokens.shape[0]):
                for r in range(reason_evidence_mask.shape[1]):
                    mask = reason_evidence_mask[b, r]
                    if bool(mask.any()):
                        token = evidence_tokens[b, mask].mean(0).detach().cpu()
                        self.tokens.append(token)
                        self.reason_ids.append(int(r))
            overflow = max(0, len(self.tokens) - self.capacity)
            if overflow:
                self.tokens = self.tokens[overflow:]
                self.reason_ids = self.reason_ids[overflow:]

    def sample_unsupported(self, target_reason: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        candidates = [t for t, r in zip(self.tokens, self.reason_ids) if r != int(target_reason)]
        if not candidates:
            return None
        rows = [candidates[i % len(candidates)].to(device=device, dtype=dtype) for i in range(batch_size)]
        return torch.stack(rows, 0)

    def __len__(self) -> int:
        return len(self.tokens)
