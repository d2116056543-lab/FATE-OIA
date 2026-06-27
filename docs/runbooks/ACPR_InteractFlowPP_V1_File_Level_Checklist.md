# ACPR-InteractFlow++ V1 file-level checklist

This checklist is subordinate to the full plan and audit skill.

## Formal package

| File | Required public objects | Dynamic evidence |
|---|---|---|
| `config.py` | typed config, unknown-key rejection, consumer manifest | config mutation report |
| `types.py` | all dataclasses in plan | one real batch shape report |
| `psi_damo_dataset.py` | PSI package loader, 15-frame direct input, Exp29 alignment | dataset contract |
| `psi_frame_resolver.py` | PSI1 frame path resolver | missing-frame hard fail test |
| `psi_metrics.py` | DAMO-compatible action/Exp29 metrics | parity fixtures |
| `visual_encoder.py` | anchor DINO + selected-layer fusion | no target-frame/no cache proof |
| `motion_path.py` | 15-frame lightweight motion tokens | temporal order test |
| `predicate_ontology.py` | OIA 32 + PSI predicates | ontology hash |
| `predicate_transfer.py` | OIA query mapper + true text embedding | source checkpoint report |
| `dynamic_predicate_field.py` | recurrent entmax predicate trajectories | gradients/order test |
| `nnpu_calalign.py` | text rules, nnPU, train-only calibration | unknown-not-negative test |
| `interaction_flow.py` | regime/phase/source/corridor states | connected factor test |
| `interaction_grammar.py` | weak state rules | grammar contradiction tests |
| `response_lag.py` | lag 0..4 | synthetic lag recovery |
| `decision_ledger.py` | exact additive logits and gates | exact identity test |
| `cluster_semantics.py` | Exp29 medoids/reliability | all-zero handling test |
| `exp29_head.py` | masked ASL and contribution alignment | positive-mask loss test |
| `interventions.py` | earliest-layer recomputation | real delta test |
| `model.py` | formal orchestration | import graph |

## Engines

| File | Required behavior |
|---|---|
| `train_acpr_interactflow_psi.py` | 30-epoch one-run training, test-only eval, atomic checkpoints |
| `eval_acpr_interactflow_psi.py` | DAMO metrics plus middle outputs |
| `profile_acpr_interactflow.py` | real 100-batch throughput/memory measurement |
| `run_acpr_interactflow_preflight.py` | orchestrate blocking gates |
| `audit_acpr_interactflow.py` | reject placeholder reports |
| `export_acpr_interactflow_visuals.py` | real Dynamic Interaction Decision Ledger |
| `build_acpr_interactflow_atlas.py` | standalone atlas |
| `supervise_acpr_interactflow_foreground.py` | attached foreground complete-suite supervisor |

## Losses

`fate_oia/losses/acpr_interactflow_losses.py` must implement:

```text
soft_action_kl
weighted_soft_action_kl
masked_asl_exp29
nnpu_loss
state_semantic_loss
response_lag_consistency
decision_ledger_residual
non_degradation_hinge
contribution_alignment_js
temporal_consistency
```

Forbidden substitutions:

```text
all-zero Exp29 negative BCE
hard CE as sole action loss
contribution magnitude residual
display-only intervention
placeholder visualization
```

## Existing files

Allowed backward-compatible modifications:

```text
fate_oia/models/acpr_dino_field.py
fate_oia/models/acpr_scene_predicate_head.py
fate_oia/transforms.py
fate_oia/losses/__init__.py
.gitignore
```

Formal trainer must not be `train_acpr_oia.py`.

## Implementation order

1. Worktree/branch safety.
2. Plan/config/skill/manifest.
3. PSI dataset and metric parity.
4. Typed contracts.
5. Visual encoder and frame resolver.
6. OIA transfer and predicates.
7. Dynamic predicate field.
8. nnPU/CalAlign.
9. Interaction-flow states and lag.
10. Decision ledger.
11. Exp29 weak explanation.
12. Losses and training loop.
13. Evaluation/best selection.
14. Interventions.
15. Visualization/atlas.
16. Throughput profile.
17. Audit/supervisor.
18. Agent B review pass.
19. Foreground formal run.
