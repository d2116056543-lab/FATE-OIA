from __future__ import annotations

from torch import Tensor

from .aie_reason_rereader import AIEReasonRereader


class PACTReasonRereader(AIEReasonRereader):
    """AIE rereader with a literal per-logit correction budget."""

    def forward(
        self,
        *args,
        reason_budget: float = 0.25,
        compatibility_mode: bool = False,
        reason_scale: float = 1.0,
        **kwargs,
    ) -> dict[str, Tensor]:
        if compatibility_mode:
            return super().forward(*args, reason_scale=reason_scale, **kwargs)
        result = super().forward(*args, reason_scale=1.0, **kwargs)
        raw_delta = result["reason_raw_delta"]
        budget = raw_delta.new_tensor(float(reason_budget)).clamp_min(0.0)
        delta = budget * raw_delta.tanh()
        primary = args[7]
        result.update(
            reason_delta=delta,
            reason_logits_final=primary + delta,
            reason_logits_final_train=primary.detach() + delta,
            reason_delta_budget=budget,
            reason_delta_to_budget_max=(delta.abs() / budget.clamp_min(1e-8)).max(),
        )
        return result
