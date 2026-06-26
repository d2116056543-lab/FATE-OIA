# ACPR-VISTA V1 Implementation Audit Skill

This worktree-local skill mirrors the user-provided ACPR-VISTA audit skill.
Formal training is forbidden unless `fate_oia.engine.audit_acpr_vista_implementation`
passes, `REVIEW_PASS_ACPR_VISTA_V1.txt` exists for the exact clean HEAD, and
VISTA gates/memory probe pass.

Hard gates:
- direct image only;
- frozen no-grad DINO ViT-S/8;
- no feature cache and no token compression;
- test-only eval and test deploy-fixed best;
- VISTA adapts selected-layer patch tokens before ego, predicate, label trunk,
  predicate reasoner, HardPair, and CalAlign;
- adapter has rank-48 low-rank path, 3x3 depthwise local geometry, zero-init
  ReZero gate, predicate-anchored patch gate floor 0.20;
- zero gate preserves ACPR-CalAlign equivalence;
- first backward gives nonzero adapter gate gradient;
- HardPair is budgeted;
- CalAlign teacher uses best-lock;
- no SECA/PMT/PACE/FusionLite/ActAlign/Candidate/MoE/selector/action-set-final.
