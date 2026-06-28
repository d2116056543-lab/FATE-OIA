from __future__ import annotations

from .acpr_threshold_head import ACPRThresholdHead


class PMCalPUCalAlignHead(ACPRThresholdHead):
    """PMCal deploy head. It preserves deploy_logits = base_logits - theta."""
