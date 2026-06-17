# ACPR-FusionLite Implementation Audit Skill

Hard gates:
- FusionLite must be zero-initialized and exactly equivalent to ACPR-CalAlign at initialization.
- No expert/MoE/specialist/selector/graph/action-set-final/cache/compression/test-threshold leakage.
- FusionLite may condition the action fusion gate using action-specific reason and predicate context only; it must not directly add predicate/action-set logits to final action.
- Audits must dynamically verify deploy = base - theta, reason logits unchanged by FusionLite, and use_fusionlite=false compatibility.
