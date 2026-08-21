# TIDA-OIA V1 Design Specification

## Scope and authority

This specification implements the user-provided TIDA plan and strict audit. The source is the exact `vetra_from_scratch_staged_v1` tree `9c885b803a34040be8d04baef81f60d6f567aa0a`. The user's latest instruction intentionally overrides the original plan's direct-branch rule: TIDA is developed in the isolated `tida_oia_v1_video` worktree and branch. No earlier VETRA worktree is modified.

The only formal method is:

`15-frame clip -> target-conditioned history read -> terminal innovation -> predicate differential local traffic state -> action/reason temporal readers`.

TIDA estimates incremental information in history conditioned on the last frame. It does not claim metric traffic flow, physical speed, calibrated TTC, or world-coordinate density.

## Data contract

The formal inventory combines exactly three audited clip batches: 3115 official-train and 885 official-test clips. Official train is deterministically partitioned by normalized source-video group into exactly 312 `train_calib`, exactly 512 `train_audit`, and 2291 `train_core` clips. Partition seed is fixed at `20260821`. Groups are ordered by `(SHA256("20260821:" + normalized_source_id), normalized_source_id)`. A 0/1 subset-sum dynamic program chooses an exact total of 312; ties choose the lexicographically smallest sorted group-index tuple. After removing those groups, the identical algorithm chooses exactly 512; all remaining groups form train_core. Inability to realize exact counts fails closed. Only `train_core` labels drive optimizer updates. `train_calib` fits per-epoch thresholds. `train_audit` provides independent mechanism diagnostics. Stage-C nested OOF uses `train_calib + train_audit`; none of these partitions may share normalized source identity, clip SHA, or endpoint near-duplicate with another partition. Official test remains 885 clips and is used only for metrics/checkpoint selection.

Every manifest row is keyed by `(official_split, file_name)`. `official_split` is exactly `train` or `test`; `partition` is exactly `train_core`, `train_calib`, `train_audit`, or `test`, with `official_split=test` iff `partition=test`. The row also carries the original BDD-OIA target image, clip path, source video identity, duration/FPS/frame count, target timestamp/index, and 4/21 multi-label targets. Duplicate keys, ambiguous mapping, missing labels, cross-partition source-video overlap, undecodable clips, and target mismatch fail closed.

The dataset returns target image `[3,360,640]`, 14 history frames `[14,3,192,344]`, 15 real timestamps ending at zero, frame indices and validity mask, action `[4]`, reason `[21]`, and provenance fields. Sampling follows `t_i=-5(1-i/14)^2`; train jitter is bounded to 10% of adjacent spacing and cannot reorder samples, while test is deterministic. Geometry and flip are synchronized across all 15 frames; target and context have separate resolutions but share the same sampled transform parameters.

The original BDD-OIA image is the formal target frame. The decoded video last frame is audit-only. After deterministic RGB decode and common-size comparison, every sample must satisfy SSIM >= 0.90, PSNR >= 20 dB, normalized MAE <= 0.08, and 64-bit pHash Hamming distance <= 16; otherwise the row is rejected and formal count mismatch fails the run. The aggregate still must satisfy the original median SSIM >= 0.995 contract. Exact clip SHA collisions across partitions fail. Normalized source-stem collisions fail. Endpoint pHash distance <= 4 with duration difference <= 0.1 s and FPS difference <= 0.5 is a deterministic near-duplicate; cross-partition occurrences are quarantined and make the formal audit fail rather than being silently retained.

## Frozen image base and shared DINO

The target frame uses the unchanged AIE/VETRA path and a supplied same-project Stage-B model checkpoint. All image-model parameters remain frozen. Target and context use the same Python DINO backbone object; no second backbone, detector, tracker, optical flow, VLM, feature cache, decoded-frame cache, or token compression is allowed.

`ACPRDinoFieldExtractor.forward()` remains numerically identical for 360x640. `forward_at_resolution()` adds dynamic grids: 360x640 -> 45x80/3600 patches and 192x344 -> 24x43/1032 patches. `ACPREgoRegionEncoder` receives `grid_hw` explicitly while preserving its old default behavior.

The context encoder holds a non-registering weak reference to `image_model.foundation.dino`; it does not assign that module as a child. Thus object identity is shared while DINO appears under exactly one state-dict namespace and one frozen owner. Checkpoint, EMA, and optimizer code reject duplicate parameter identities.

History DINO runs in no-grad chunks. Each chunk is immediately aggregated by target-conditioned attention; this required query aggregation is not the forbidden precomputed token compression. Full `[B,14,3,1032,384]` history fields are never retained.

Before modifying the shared DINO wrapper, the source worktree and supplied image checkpoint produce a golden-oracle artifact over 16 fixed real test images in eval mode, fp32, no augmentation, `action_scale=1`, and `reason_scale=1`. It stores input/checkpoint/source hashes and complete tensors for DINO selected-layer CLS/patches; `action_logits_primary`, `reason_logits_primary`; `predicate_logits`, `predicate_probs`, `predicate_tokens`, `predicate_attention`, `predicate_layer_weights`; `action_nodes_primary`, `reason_nodes_primary`; action evidence token/map/reference/sampling offsets/sampling weights/layer mixture; bounded action contribution/final action; reason private evidence/route/delta/final reason; and branch logits. Source-vs-TIDA fallback tolerances are max-abs <1e-6 fp32 and <5e-4 bf16 for every listed tensor. TIDA fallback is also compared to its runtime image branch, preventing two equally broken wrappers from passing a relative comparison.

## Target-conditioned history read

Each clip uses 36 queries: four detached target action nodes plus 32 `LN(predicate identity + detached target predicate token)` queries. DINO layers are read recursively in semantic order 11 -> 7 -> 3 through independent normalized Q/K/V projections and bounded residual gains. Entmax-1.5 produces sparse spatial attention. The reader returns `[B,14,36,384]` tokens, `[B,14,36,1032]` attention, five-region predicate masses, and layer update diagnostics.

No reason logits, reason labels, text embeddings, CLS averaging, or pre-read layer mean enters this reader.

## Temporal encoding and innovation

A two-layer, four-head, causal Transformer encodes each of the 36 trajectories using continuous relative timestamps and frame-valid key masks. All-invalid history returns a zero summary.

One shared terminal predictor runs twice: with history and with a null history. Its target is the detached terminal action/predicate evidence; the evidence being predicted never enters predictor input. Per-query error is equal-weight normalized Huber plus cosine distance. Reliability and innovation are exactly:

`rho = stopgrad(clamp((e0-eH)/(e0+eps), 0, 1))`

`Xi = rho * LN(EhatH-Ehat0)`.

All-invalid history forces rho and Xi to zero. Gain loss detaches `e0`; no-history reconstruction remains independently trained. Real order must be distinguished from repeated-last, shuffled, and reversed history.

## Predicate differential state

The 32 current predicate names are classified exactly once as `static_anchor`, `dynamic_actor`, or `terminal_context`. First and second derivatives use actual timestamps. A detached robust center over static-anchor velocity removes common camera tendency; it is explicitly not optical flow or ego pose.

Each predicate state combines terminal token, predicate innovation, EMA relative velocity/acceleration, region-mass derivatives, persistence, and rho. Region masses always use the runtime grid. Human-facing dynamic concepts are deterministic, non-trainable translations with an `unknown` fallback and never become pseudo targets or action rules.

## Action and reason readers

The action bank contains exactly 32 predicate differential factors, four action innovation factors, and one null factor. Keys may use identities, roles, and reliability. Values are visual only. Entmax-1.5 routing includes `log(rho+eps)`. The bounded action residual uses kappa 0.15. Let `s=sum(c)` and `delta=kappa*tanh(s/kappa)`. Contributions use `c*(delta/s)` when `|s|>eps`; the analytic limit `c` is used at zero, where `delta=0`, followed only by a floating-point residual correction assigned to the largest-absolute contribution. This is numerical reconciliation, not a learned bias. Their fp32 sum must equal `video_action-image_action` within 1e-6. Null-only, zero-rho, invalid-history, disabled-history, or zero temporal scale returns the image logits exactly.

The reason reader is private. Predicate state and selected action temporal evidence are explicitly detached before private cross-attention. The reason residual uses kappa 0.12. Reason loss may update only the private reason reader; it cannot update image base, history reader, temporal encoder, innovation predictor, predicate differential shared path, or action reader. Per-label deployment beta is selected only by nested OOF on train_calib plus train_audit.

## Loss and ownership

Trainable owners are exactly: history reader, temporal encoder, innovation predictor, predicate differential projection, temporal action, and temporal reason. Every parameter belongs to one owner and one optimizer group.

`L_no_history` is the unweighted distance `d(Ehat0,E)`; the `0.25` coefficient is applied exactly once in the innovation aggregate. Innovation loss is `L_hist + .25 L_no_history + .20 L_gain + .10 L_order + .10 L_repeat`. Action loss is `1.00 ASL + .15 SmoothAP + .10 base_protect + .005 delta + .005 route_sparse`, macro-averaged across four actions. Reason loss is `1.00 partial/ASL + .08 rank + .04 softF1 + .005 delta`. Total is `.25 innovation + action + reason`. Every configured term is computed and added exactly once; unavailable diagnostics carry `available=false` and a reason, never a fabricated zero pass.

During the first 5% of updates the formal video logits use zero temporal scale, but the unscaled route is still trained through `route_sparse`. For valid clips with `rho_max>0`, define `H(pi)=-sum(pi*log(pi+eps))`, non-null mass `m=1-pi_null`, and action-route centroids `u_a=sum_f pi_af * normalize(k_f)`. The scalar term is `mean(H(pi)/log(37)) + mean(relu(0.05-m)) + mean_{a!=b} relu(cos(u_a,u_b)-0.90)`. Differential factor keys are computed from `predicate_differential_state` without detach, so this loss has a mandatory autograd path into the differential projection and action key projection. A synthetic valid-rho probe must show both gradients >0 while final video logits equal image logits under zero scale. The innovation terms train the history reader, temporal encoder, and predictor. Private reason updates begin only when its scale ramps above zero. A phase-aware owner/gradient matrix is asserted dynamically.

## Training, evaluation, deployment

The frozen image checkpoint and clip manifest are mandatory CLI inputs. Ten epochs use BF16 AdamW, owner-specific learning rates, 5% update warmup, cosine scaling of temporal residuals from 5-20%, and full strength thereafter. EMA covers only TIDA trainable parameters. No metric/patience/target early stop is allowed.

Every epoch fits deployment parameters on train_calib only, then evaluates the sole evaluation split, test, once. `train_calib` is a parameter-fitting partition, not an evaluation split. The run reports image/online-video/EMA-video raw and deploy metrics, saves every epoch checkpoint plus named best checkpoints, and labels the protocol `internal_test_selected=true`, `publication_eligible=false`. Test labels are explicitly allowed only for metric computation and the user-requested checkpoint selection; they never fit thresholds, TTA weights, reason beta, model parameters, learning-rate changes, or structural decisions.

Stage C uses synchronized original/flip clips, explicit left/right remapping, and nested OOF on train_calib plus train_audit. It emits a self-contained deployment checkpoint and manifest.

The foreground PowerShell entry invokes the Python supervisor synchronously with inherited stdout/stderr. Structural failures fail closed with artifacts; weak metrics never stop a healthy run. Exact resume restores model, optimizer, scheduler, EMA, epoch/update, all RNG state, sampler state, hashes, and best-source metadata.

## Verification gates

Verification includes RED-first unit tests, legacy regression, exact image fallback against an independent source-tree oracle, shared DINO identity, dynamic grids, chunk retention hooks, causal/time/mask interventions, innovation formula and stopgrad, differential derivatives, contribution reconstruction, reason gradient firewall, loss owner/once registry, real-video interventions, 2+2 update exact resume, artifact schema, foreground-only scan, 4000-row data audit, real-video smoke, and real-video memory profile.

Cross-split leakage detection uses normalized source IDs, clip content SHA256, file metadata, endpoint perceptual hashes, and near-duplicate endpoint comparisons; a string-ID comparison alone is insufficient. Each sample must pass an individual target-frame floor in addition to aggregate distribution thresholds. `repeated_last` repeats the last valid history query token, never the target-frame token.

Each memory candidate runs 10 warm-up microsteps and 50 measured microsteps, exactly matching the supplied plan and strict Skill. Reserved-memory growth is a fitted slope normalized to 100 measured microsteps and is labeled as such; it is not misreported as 100 optimizer updates. This resolves the source documents' microstep/update wording conflict without reducing the measured candidate procedure.

Audit trust is evidence based: the PASS writer accepts no caller-supplied pass booleans. It recomputes from raw tensors, hashes, command lines, exit codes, hook counts, and test reports. Negative mutation tests must demonstrate that each principal audit gate fails when its invariant is deliberately violated.

No full training is allowed until clean-HEAD artifacts bind plan, skill, config, clip manifest, image checkpoint, Git head/tree, test results, and all four review passes into `FULL_TRAIN_READY_TIDA_OIA_V1.json`.
