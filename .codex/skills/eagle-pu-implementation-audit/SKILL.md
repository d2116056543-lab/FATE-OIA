---
name: eagle-pu-implementation-audit
description: Hard audit for EAGLE-PU V1 FATE-OIA implementation.
---

# EAGLE-PU Implementation Audit

Use only for `eagle_pu_v1_direct_image`. Hard gates:
- no RunC checkpoint, no cached logits, no feature cache, no token compression, no val eval/best
- direct image -> frozen DINO -> state bank -> label trunk -> prototype/graph reason delta -> raw/calibrated outputs
- action final raw must equal direct action path; action-set auxiliary must never feed final action
- audit must run `fate_oia.engine.audit_eagle_pu_implementation --write_review_pass`
- full train must not start without `.background_runs/eagle_pu_v1_preflight/REVIEW_PASS_EAGLE_PU_V1.txt`
