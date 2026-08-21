# TIDA-OIA V1 Implementation and Verification Plan

## Deliberate plan deviation

The original supplied plan required direct modification of `vetra_from_scratch_staged_v1`. The user explicitly superseded that rule on 2026-08-21 by requiring a new worktree. The remote source branch had advanced without tree-content changes, so implementation uses branch `tida_oia_v1_video` from exact remote base commit `cfeb25f09ea4452decf9326990f02d01895926e0` and tree `9c885b803a34040be8d04baef81f60d6f567aa0a`. This is the only intentional deviation. Branch names, Git binding checks, and audit expectations are updated accordingly; all method and evidence requirements are preserved.

Local worktree: `C:\Users\WLJTXY\Documents\Codex\2026-04-26-tailscale-100-127-14-110-vscode\.tmp_tida_oia_v1`. Remote worktree after code push: `E:\sbw\FATE_Drive\fate_oia_tida_oia_v1_worktree`. Remote ref: `refs/heads/tida_oia_v1_video`. The source branch is read-only; all pushes target only the new ref.

The command contract is `git -C <source> worktree add -b tida_oia_v1_video <new-path> cfeb25f...`; push is only `git push origin HEAD:refs/heads/tida_oia_v1_video`. Before and after implementation, record source worktree HEAD/tree/status and reject any change. Never push TIDA commits to `vetra_from_scratch_staged_v1`.

## Resolved plan ambiguities

- `L_no_history` means raw distance and receives coefficient 0.25 once, resolving the section 9/15 double-weight conflict.
- Exact factor contribution uses the analytic `delta/s` scale plus a non-learned floating-point residual correction; the original `s+eps` approximation is not used as an equality claim.
- The shared DINO is weak-referenced by context code and registered once under the image model.
- Query aggregation is required online attention reduction, not prohibited cached/precomputed token compression.
- First-5% differential gradient comes from an active route sparsity/non-null/diversity term over the unscaled route; final temporal logits remain exactly zero-scaled.
- `train_calib` fitting may run every epoch, but the only evaluation split is test. Test labels are permitted for metrics and explicit checkpoint selection only.
- Memory candidates use 10 warm-up plus 50 measured microsteps as required by the supplied plan and Skill; reserved-memory slope is normalized and explicitly labeled per 100 measured microsteps.
- `repeated_last` repeats the final valid history representation, never target evidence.
- The required files include `tida_artifacts.py` and `tida_contracts.py`; completion artifacts carry full schemas and hashes.

## Coverage matrix

| Must-have requirement | Implementation surface | Verification evidence | Status |
|---|---|---|---|
| Preserve old VETRA and use isolated worktree | Git worktree/branch | source HEAD/tree, worktree list, legacy regression | in progress |
| Canonical 3115/885 video inventory | manifest builder and data audit | exact counts, cross-batch duplicate/split/source checks | planned |
| Cross-split aliases/near duplicates | manifest/data audit | SHA256 + endpoint pHash/metadata collision artifact | planned |
| Unique image-to-clip mapping | `tida_clip_manifest.py` | ambiguous/missing negative tests | planned |
| 15-frame quadratic sampling and bounded jitter | `bdd_oia_video.py` | exact formula/order/determinism tests | planned |
| Synchronized target/history augmentation | `transforms_video.py` | shared flip/letterbox mapping tests | planned |
| Original image as target plus decoded-last audit | dataset/data audit | SSIM/PSNR/NMAE/pHash artifacts | planned |
| Dynamic DINO resolutions without changing old forward | `acpr_dino_field.py` | old-vs-new equivalence and shape tests | planned |
| Independent source-tree image oracle | pre-change source worktree artifact | real input/tensor hashes and exact comparison | planned |
| Dynamic ego grid preserving old default | `acpr_ego_regions.py` | 3600/1032 row tests and old-path equivalence | planned |
| One frozen DINO object | context encoder/model | identity/state-dict/grad/hook audit | planned |
| Chunk history and release dense patches | context encoder | peak-shape hook and memory tests | planned |
| 4 action + 32 predicate target queries | terminal query reader | provenance/shape/identity tests | planned |
| Recursive 11->7->3 independent entmax read | terminal query reader | layer intervention/order/no-mean tests | planned |
| Continuous-time causal temporal encoder | temporal encoder | timestamp, future, invalid/all-invalid tests | planned |
| Shared history/no-history terminal predictor | terminal innovation | module identity/call count tests | planned |
| No terminal target leakage | terminal innovation/model | input provenance hooks | planned |
| Exact rho/Xi formula and stopgrad | terminal innovation/loss | recomputation and gradient tests | planned |
| Real-order/repeat/shuffle identifiability | loss/interventions | nonzero losses and real-video audit | planned |
| Predicate role exact cover by name | role YAML/differential | coverage test against runtime names | planned |
| Timestamp derivatives and robust common motion removal | differential | analytic trajectory tests | planned |
| Runtime-grid region mass | differential | 24x43/45x80 tests | planned |
| Non-gradient deterministic concepts with unknown | explain module | no-parameter/no-grad/confidence tests | planned |
| 37-factor visual-only action bank | action reader | provenance and bank-shape tests | planned |
| Sparse target-specific route with null | action reader | sum/zeros/nonuniform/4-action coverage tests | planned |
| Bound 0.15 and exact contribution sum | action reader/model | fp32 1e-6 reconstruction test | planned |
| Exact image fallback under all five conditions | model | target equivalence matrix | planned |
| Private reason reader bound 0.12 | reason reader | shape/bound/nonzero-private-grad tests | planned |
| Explicit reason detach firewall | model/loss registry | owner gradient tomography | planned |
| Train-only reason beta | deployment exporter | no-test-label mutation test | planned |
| All 14 loss terms exactly once | losses/registry | registry and synthetic violation tests | planned |
| Owner exact cover | loss registry/trainer | parameter-id set equality | planned |
| Formal intervention suite without target rerun | utility/evaluator | hook counts and 128-clip artifact | planned |
| Frozen supplied Stage-B image checkpoint | trainer/contracts | lineage/hash/frozen-state tests | planned |
| Update-ratio schedule, BF16 AdamW, TIDA-only EMA | config/trainer | resolved config and update probes | planned |
| First-5% differential has legal gradient | route regularizer/owner matrix | nonzero legal gradient with zero final scales | planned |
| Every epoch test-only evaluation and all checkpoints | trainer/evaluator | epoch artifact/checkpoint schema | planned |
| Train-only calibration and Stage C | collector/exporter | leakage tests and OOF provenance | planned |
| Exact checkpoint/resume | artifacts/trainer | 4 vs 2+resume+2 equality test | planned |
| Real-video memory candidate profile | profiler | A-D profile, <=45 GiB, leak slope | planned |
| Foreground-only supervisor | supervisor/PowerShell | forbidden launch scan and child-exit tests | planned |
| Strict audit cannot self-certify | audit engine/skill | dynamic probes plus negative mutation tests | planned |
| Clean HEAD equals GitHub branch before training | Git binding | local/remote SHA artifact | planned |
| Ten-epoch foreground training and Stage C | supervisor | 10 metrics/checkpoints, deployment, completion JSON | planned |
| Append canonical three MD files only | execution log | matching remote/local hashes and append markers | planned |

## TDD execution order

1. Add the full `tests/test_tida_*.py` contract suite as RED tests, grouped by data, frozen-image/DINO, temporal mechanism, ownership/loss, runtime protocol, and artifacts.
2. Implement manifest schema, multi-rate sampling, video decode, synchronized transforms, and strict data audit. Validate all 4000 rows and fail closed on cross-batch conflicts.
3. Extend DINO and ego grid interfaces while proving old 360x640 output equality. Add shared, chunked context encoding.
4. Implement sparse query reader and continuous-time causal encoder. Verify layer order, target conditioning, no reason input, and all masks.
5. Implement shared terminal predictor, exact reliability/innovation formula, and identifiability losses.
6. Implement role-bound predicate differential state and deterministic dynamic concept export.
7. Implement visual-only action route/contribution and detached private reason route. Run full owner-gradient firewall tests.
8. Integrate `TIDAOIAModel` and formal interventions; verify exact fallback and no target DINO rerun.
9. Implement the loss registry, trainer, evaluator, checkpoint/EMA/resume, train-only deployment fitting, profiler, foreground supervisor, config, and structured artifacts.
10. Implement the audit engine against the already-installed and frozen strict Skill. The Skill must not change after DESIGN_REVIEW; any later Skill edit invalidates the design pass and requires a new supervision review and design-pass hash.
11. Run compile, targeted tests, selected AIE/VETRA regressions, 4000-row data audit, real-video smoke, memory profile, exact resume, and clean-HEAD re-audit.
12. Commit/push code-only changes, bind review artifacts to the final clean HEAD, generate `FULL_TRAIN_READY_TIDA_OIA_V1.json`, then run all ten epochs in the attached foreground supervisor and export Stage C.

## Phase-aware gradient matrix

| Backward source | image base | history reader | temporal encoder | innovation | differential | action | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| innovation terms | 0 | >0 | >0 | >0 | 0 | 0 | 0 |
| route regularizer at 0-5% | 0 | allowed | allowed | allowed | >0 | >0 | 0 |
| action terms after 5% | 0 | allowed | allowed | allowed | >0 | >0 | 0 |
| reason terms | 0 | 0 | 0 | 0 | 0 | 0 | >0 |

The trainer asserts these owner relations using parameter identities; no optimizer-group omission is accepted as a substitute for explicit detach.

## Call-graph evidence

The formal PowerShell -> supervisor -> trainer -> model -> context -> query -> temporal -> innovation -> differential -> action -> reason -> loss -> backward -> evaluator -> artifact chain is checked by static imports, runtime hook counts, and downstream tensor consumption. Audit-only outputs are explicitly listed and cannot satisfy a formal-output gate. Any formal tensor with zero call count, no consumer, or fixed value fails.

## Completion artifact schemas

All PASS artifacts share Git head/tree, base/source head/tree, plan/skill/spec hashes, command/exit records, gates, warnings, and timestamp. Stage-specific required fields are:

- Design PASS: config/manifest/image checkpoint/test/raw-tensor hashes are null with `not_yet_produced`; design/spec/plan/skill/Git fields are required.
- Implementation PASS: config and test-report hashes are required; manifest/image-checkpoint/raw-tensor hashes may be null with `awaiting_mechanism`.
- Mechanism PASS: manifest, image-checkpoint, golden-oracle, raw-tensor, data-audit, and smoke hashes are required.
- Memory PASS: all mechanism fields plus profile report and selected candidate are required.
- Full-train-ready: every field is non-null, all prerequisite PASS hashes are embedded, tree is clean, and local/remote new-branch heads match.

`TRAIN_COMPLETED_TIDA_OIA_V1.json` additionally contains ten epoch identities, checkpoint hashes, best source/view, Stage-C deployment hash, metrics paths, training command, start/end times, and code/config immutability proof.

## Required files

Production files: `configs/fate_oia_train_tida_oia_v1_15f.yaml`, `configs/tida_predicate_roles.yaml`, `fate_oia/datasets/bdd_oia_video.py`, `fate_oia/datasets/tida_clip_manifest.py`, `fate_oia/transforms_video.py`, `fate_oia/models/tida_context_encoder.py`, `tida_terminal_query_reader.py`, `tida_temporal_encoder.py`, `tida_terminal_innovation.py`, `tida_predicate_differential.py`, `tida_action_reader.py`, `tida_reason_reader.py`, `tida_oia_model.py`, `fate_oia/losses/tida_losses.py`, `tida_loss_registry.py`, `fate_oia/utils/tida_temporal_interventions.py`, `tida_artifacts.py`, `tida_contracts.py`, `fate_oia/explain/tida_dynamic_concepts.py`, `fate_oia/engine/build_tida_clip_manifest.py`, `audit_tida_video_data.py`, `train_tida_oia.py`, `evaluate_tida_oia.py`, `profile_tida_oia.py`, `audit_tida_oia_implementation.py`, `collect_tida_tta_outputs.py`, `export_tida_deployment.py`, `supervise_tida_oia_foreground.py`, and `scripts/FATE_OIA_tida_oia_v1_foreground.ps1`.

Contract tests: `test_tida_clip_manifest.py`, `test_tida_video_dataset.py`, `test_tida_multirate_sampler.py`, `test_tida_synchronized_transform.py`, `test_tida_last_frame_contract.py`, `test_tida_split_leakage.py`, `test_tida_dynamic_dino_grid.py`, `test_tida_target_frame_equivalence.py`, `test_tida_shared_dino_identity.py`, `test_tida_no_duplicate_backbone.py`, `test_tida_dino_frozen.py`, `test_tida_context_chunking.py`, `test_tida_query_reader_shapes.py`, `test_tida_query_reader_layer_order.py`, `test_tida_query_reader_uses_target_queries.py`, `test_tida_no_layer_mean_before_read.py`, `test_tida_temporal_causal_mask.py`, `test_tida_continuous_timestamp.py`, `test_tida_invalid_frame_mask.py`, `test_tida_shared_terminal_predictor.py`, `test_tida_no_target_leak_into_predictor.py`, `test_tida_innovation_formula.py`, `test_tida_rho_stopgrad.py`, `test_tida_no_history_fallback.py`, `test_tida_repeated_last_low_innovation.py`, `test_tida_predicate_role_exact_cover.py`, `test_tida_time_derivatives.py`, `test_tida_common_motion_removal.py`, `test_tida_region_mass.py`, `test_tida_dynamic_concepts_nongrad.py`, `test_tida_action_visual_value_only.py`, `test_tida_action_null_fallback.py`, `test_tida_action_contribution_exact_sum.py`, `test_tida_action_residual_bound.py`, `test_tida_reason_firewall.py`, `test_tida_reason_private_nonzero_grad.py`, `test_tida_reason_beta_train_only.py`, `test_tida_loss_owner_exact_cover.py`, `test_tida_loss_terms_once.py`, `test_tida_no_placeholder_losses.py`, `test_tida_temporal_interventions.py`, `test_tida_same_terminal_flatten.py`, `test_tida_no_target_dino_rerun_for_audit.py`, `test_tida_memory_contract.py`, `test_tida_resume_exact.py`, `test_tida_test_every_epoch.py`, `test_tida_test_selected_best.py`, `test_tida_foreground_only.py`, `test_tida_no_metric_early_stop.py`, `test_tida_artifact_schema.py`, and `test_tida_git_head_binding.py`.

Docs/Skill: this spec, this implementation plan, and `.codex/skills/tida-oia-v1-implementation-audit/SKILL.md`. Existing single-frame entrypoints remain backward compatible.

## Completion rule

No broad `pass=true` artifact is accepted. Completion requires fresh command output and hash-bound design, implementation, mechanism, memory, Git, data, resume, and foreground-supervisor evidence. The only pre-training decision text is `APPROVED_FOR_FULL_TRAIN`; otherwise it is `REJECTED` with concrete missing items.
