# COEV-OIA V1 Epoch-Budget Amendment

This amendment supersedes only the original frame-count, epoch-count, and
per-epoch sampling/audit-budget clauses. All model, data-integrity, no-cache,
no-compression, real-RGB, test-only, gradient, intervention, and completion
contracts remain mandatory.

## Fixed Training Contract

- Keep the complete 15,302-record training pool.
- Draw 6,400 unique clips per epoch for 18 epochs; never create a permanent
  fixed 6,400-record subset.
- Draw exactly 3,200 dynamic, 1,920 balanced, and 1,280 static/control clips.
- Build novelty strata from clips with valid history. Reserve 320 of the 1,280
  static/control slots for rotating no-history image controls and use the
  remaining 960 for genuine low-dynamic videos. This keeps every training row
  in rotation without letting unavailable history masquerade as low motion.
- Novelty strata use only image-derived temporal metadata. Action and reason
  labels may balance coverage within a stratum but may not define novelty.
- Prefer the least-exposed records in every stratum before avoidable repeats.
  The sampler state, exposure counts, permutation, metadata SHA-256, and
  consumed position are checkpointed for exact resume.
- Use nine frames spanning the original five seconds: eight quadratic history
  samples plus the endpoint. Motion, coupling, and state-change operators use
  decoded actual timestamps, never ideal requested timestamps.
- Keep effective batch 32, 200 optimizer updates per epoch, 3,600 total
  updates, 180 warmup updates, and the existing warmup-cosine schedule.
- Run the normal full 4,572-row test forward once per epoch. Run expensive
  traffic interventions on a fixed 512-row test-audit subset per epoch and on
  the complete test set once for the final best checkpoint.

## Required Evidence

- `sampler_epoch_stats.jsonl` records exact quotas, uniqueness, label-positive
  counts, prior exposure, pool coverage, and history availability overall and
  by stratum.
- Novelty metadata contains no Action/Reason fields and exactly one row for
  every training record. Decode errors fail preflight.
- Nine-frame real-RGB smoke proves dynamic tensor shapes and full-history KV
  reread length. A stale 15-frame constant must not determine PASS.
- Runtime profiling reports train throughput and separately measures normal
  full-test and 512-row intervention costs before formal launch.

This is a compute-allocation change, not a claim that nine frames or the
stratified sampler necessarily improve test metrics. Empirical improvement is
reported only from the formal fixed-threshold test outputs.
