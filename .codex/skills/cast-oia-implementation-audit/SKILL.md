---
name: cast-oia-implementation-audit
description: Audit CAST-OIA V1 for exact GPTPro plan compliance before training.
---

# CAST-OIA Implementation Audit

Hard gates:
- action-set exactness
- combo loss anti-collapse
- label-specific sparse evidence
- text grounding
- ego-coordinate use
- evidence graph
- reason reliability
- full model forward
- train protocol
- foreground supervisor

Reject RunC, cached logits, feature cache, val-best, token compression, background process launch, and softmax(action_logits).
Require real-DINO smoke and REVIEW_PASS bound to the current git HEAD before full training.
