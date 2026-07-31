# METER-OIA V3 / HECA Implementation Audit

This audit is fail-closed. Passing means the code implements and invokes HECA; it does not guarantee metric gains.

## Hard Gates

- Read the canonical `task_plan.md`, `findings.md`, and `progress.md` before remote work.
- Require direct RGB image, frozen DINO ViT-S/8 layers 3/7/11, full 3600 tokens, no cache/compression/RunC.
- Require typed semantic/spatial/state measurement, offline ontology prototypes, train-main factor-specific tau, null partition, and source-weighted anchor/state supervision.
- Reject hard action-factor masks and sample admission. Require learned all-action allocation, state-conditioned values, exact additive contribution conservation, and 5% selective state/global bridge.
- Require CalAlign-anchored global reason, detached evidence correction, observed-positive weight 1, one noisy-zero trust object, private-only PU, and geometric/light view consistency.
- Require shared/private rank-16 adapters, shared-only excess-risk gradient ownership, adaptive foundation cap/LR, one corruption per optimizer update, and exact resume state.
- Require cheap diagnostics to reuse one DINO field. B0-B5 must be independent runs, never aliases for same-forward branches.
- Reject weak BDD100K targets in test forward and test metrics in threshold, PU, LR, or gate updates.
- Require all `test_heca_*.py`, py_compile, V3 implementation audit, real-DINO profile, and real 4-epoch pilot Gates A-G.
- Require the trainer and supervisor to recompute A-G from exactly four epochs of raw branch/typed/runtime/loss evidence plus the bound audit/ontology/tau/gradient inputs. A self-reported or self-hashed Gate C file must never unlock full training.
- REVIEW_PASS is code/preflight-only. Full training additionally requires clean matching HEAD plus a matching pilot pass and Gate C artifact.

## Gate C Boundary

Pilot correction fraction is 0.20. Full correction fraction is 0.25 only after Gate C passes. Never weaken the gate or reuse a pilot checkpoint as a formal result.
