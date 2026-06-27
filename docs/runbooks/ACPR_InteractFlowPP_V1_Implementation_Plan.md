# ACPR-InteractFlow++ V1 for PSI DAMO-Compatible Protocol
## Codex code-level implementation, audit, experiment, and foreground-supervision contract

**Repository:** `https://github.com/d2116056543-lab/FATE-OIA`  
**Source branch:** `acpr_calalign_v1_2`  
**Source branch HEAD when this contract was written:** `373aa49feac17372574fd7fb056c1d79c7c848fe`  
**New worktree:** `E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree`  
**New branch:** `acpr_interactflow_pp_v1`  
**Formal method:** `ACPR-InteractFlow++ V1`  
**Formal config:** `configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml`  
**Formal namespace:** `fate_oia/acpr_interactflow/`  
**Formal trainer:** `python -m fate_oia.engine.train_acpr_interactflow_psi`

---

## 0. Non-negotiable objective

Implement a PSI video model that migrates the user's strongest BDD-OIA ACPR-CalAlign idea into the public PSI DAMO-compatible 11,902-sample protocol:

```text
BDD-OIA static action-inducing predicates
→ PSI dynamic action-inducing predicate trajectories
→ pedestrian–vehicle interaction-flow states
→ response-lag decision reasoning
→ exact maintain/reduce/stop decision ledger
→ contribution-aligned Exp29 weak explanation
→ state/evidence/temporal intervention proof
```

This is not a direct BDD-X caption/control method and not a new generic video classifier. The formal claim is:

> Static action-inducing predicate reasoning can be transferred from BDD-OIA image-level decision explanation to PSI video-level vehicle–pedestrian interaction decision explanation.

The final PSI protocol is the public reconstruction package:

```text
E:\sbw\PSI-DriverDecision-Prediction-main\.worktrees\damo_repro\artifacts\psi_damo_compatible_11902_verified_20260626
```

It is **not** a bit-exact private ABIM/InAction/SGDCL/DAMO processed split. This boundary must be written into all run manifests and final reports.

---

## 1. Required inheritance from the current branch

The source branch `acpr_calalign_v1_2` already contains the strongest current ACPR-OIA formal path. The new PSI branch must reuse the *ideas and proven low-level components* but must not accidentally keep the old BDD-OIA task semantics.

Reusable low-level pieces:

```text
fate_oia.models.acpr_dino_field.ACPRDinoFieldExtractor
fate_oia.models.acpr_scene_predicate_head.ACPRScenePredicateHead
fate_oia.models.acpr_ego_regions.ACPREgoRegionEncoder
fate_oia.transforms.AspectRatioLetterboxTransform
ACPR-CalAlign checkpoint loading logic
ACPR threshold/calibration utilities where train-only
optimizer/scheduler/checkpoint/test-only record lessons
```

Do not use `ACPROIAModel` as the formal PSI model. It is image-level BDD-OIA specific. The formal PSI model must live under:

```text
fate_oia.acpr_interactflow.model.ACPRInteractFlowPPModel
```

and must produce typed PSI outputs.

Strict preservation from ACPR-CalAlign:

```text
OIA predicate names/order and learned query prior
true no-cache/no-token-compression discipline
test-only best selection requested by user
train-calib only for calibration/thresholds
separate base/deploy/calibrated artifact records
foreground supervision discipline
```

Strict changes for PSI:

```text
3-class soft driving decision instead of BDD-OIA 4/5 multi-hot action
29 weak Exp cluster labels instead of BDD-OIA 21 reason labels
15-frame temporal window instead of one image
interaction-flow states instead of image-only reason logits
state/evidence/temporal intervention as mandatory proof
```

---

## 2. Repository/worktree procedure

### 2.1 Read durable context before anything

Codex must read these first, before Git operations or code edits:

```text
E:\sbw\FATE_Drive\task_plan.md
E:\sbw\FATE_Drive\findings.md
E:\sbw\FATE_Drive\progress.md

E:\sbw\PSI-DriverDecision-Prediction-main\.worktrees\damo_repro\artifacts\psi_damo_compatible_11902_verified_20260626\PSI_DAMO_COMPATIBLE_DATASET_REPORT.txt
E:\sbw\PSI-DriverDecision-Prediction-main\.worktrees\damo_repro\artifacts\psi_damo_compatible_11902_verified_20260626\manifest.json
E:\sbw\PSI-DriverDecision-Prediction-main\.worktrees\damo_repro\artifacts\psi_damo_compatible_11902_verified_20260626\PSI_DAMO_COMPATIBLE_DATASET_FULL_DESCRIPTION_20260626.txt
```

The training/experiment status record remains only in the three canonical root files. This implementation plan and audit skill are runbook artifacts, not experiment-status files.

### 2.2 Create a new worktree from the current pushed source branch

Use the installed Superpowers worktree skill.

PowerShell:

```powershell
$Root = "E:\sbw\FATE_Drive"
$Source = "$Root\fate_oia_acpr_calalign_v1_2_worktree"
$Target = "$Root\fate_oia_acpr_interactflow_pp_worktree"
$SourceBranch = "acpr_calalign_v1_2"
$TargetBranch = "acpr_interactflow_pp_v1"

Set-Location $Source
git branch --show-current
git status --short
git fetch github
git rev-parse HEAD
git ls-remote github "refs/heads/$SourceBranch"
```

Required before creating the new worktree:

```text
current source branch == acpr_calalign_v1_2
source worktree clean
source local HEAD == github/acpr_calalign_v1_2 HEAD
```

If source is dirty:

1. inspect all changed/untracked files;
2. save patch under ignored `.background_runs/pre_interactflow_snapshot/`;
3. separate source changes from run artifacts;
4. run relevant tests;
5. commit a safety snapshot to `acpr_calalign_v1_2`;
6. push to `github/acpr_calalign_v1_2`;
7. verify local and remote SHA equality.

Forbidden:

```text
git reset --hard
git clean -fd
discarding unknown edits
copying an uncommitted worktree as reproducible source
```

Create target:

```powershell
$BaseSha = (git -C $Source rev-parse HEAD).Trim()
if (Test-Path $Target) { throw "Target worktree already exists; inspect instead of overwriting." }

git -C $Source worktree add -b $TargetBranch $Target $BaseSha
git -C $Target push -u github "$TargetBranch`:$TargetBranch"

git -C $Target rev-parse HEAD
git -C $Target status --short
git -C $Target ls-remote github "refs/heads/$TargetBranch"
```

Write:

```text
.background_runs/acpr_interactflow_pp_v1_preflight/worktree_provenance.json
```

containing source branch/SHA, target branch/SHA, worktree paths, timestamp, and remote equality proof.

---

## 3. Superpowers workflow

Use the installed equivalents of:

```text
using-git-worktrees
brainstorming
writing-plans
test-driven-development
systematic-debugging
executing-plans
requesting-code-review
receiving-code-review
verification-before-completion
```

Use two roles.

### Agent A — implementer

Agent A must:

- write an implementation manifest mapping every section of this plan to code, tests, and dynamic evidence;
- write failing tests before each component;
- implement the formal PSI namespace;
- expose typed outputs, gradients, interventions, throughput, and visualization artifacts;
- never issue the training authorization pass.

### Agent B — independent adversarial reviewer

Agent B must:

- start from this plan and the audit skill, not Agent A's summary;
- inspect the actual import graph and runtime traces;
- run all blocking gates;
- reject placeholder reports, orphan YAML fields, old BDD-OIA semantics, all-zero Exp29-as-negative supervision, dead gradients, fake interventions, and fake DAMO metric parity;
- issue the review pass only for the exact clean pushed target SHA.

Formal loop:

```text
Agent A plan
→ Agent B plan review
→ Agent A implementation with TDD
→ targeted tests
→ real data smoke
→ mechanism/throughput/intervention gates
→ Agent B full audit
→ blocker fixes with regression tests
→ commit/push
→ audit rerun on exact clean pushed SHA
→ review pass
→ foreground formal training
```

---

## 4. PSI protocol contract

### 4.1 Dataset files

Use:

```text
dataset_package_root:
  E:\sbw\PSI-DriverDecision-Prediction-main\.worktrees\damo_repro\artifacts\psi_damo_compatible_11902_verified_20260626

samples:
  samples/train.pkl
  samples/val.pkl
  samples/test.pkl
  samples/train.jsonl
  samples/val.jsonl
  samples/test.jsonl

reason labels:
  reason_exp29/train.pkl
  reason_exp29/val.pkl
  reason_exp29/test.pkl

label embeddings:
  label_embedding/label_embedding_psi_exp29.pkl
  label_embedding/label_embedding_psi_exp29.json
```

Raw frame roots:

```text
PSI1:
  E:\sbw\PSI-DriverDecision-Prediction-main\PSI_Dataset_1_2\PSI\PSI1.0

PSI2 full root may be present but is not the formal source:
  E:\sbw\PSI-DriverDecision-Prediction-main\PSI_data
```

### 4.2 Dataset boundary

Every manifest must state:

```text
This is a public reconstruction of the PSI-1.0 DAMO/ABIM/InAction-style 11,902-sample protocol,
not a bit-exact private ABIM/InAction/SGDCL/DAMO processed split.
```

### 4.3 Split and labels

Expected exact counts:

```text
train rows = 8873, videos = 82
val rows   = 612,  videos = 6
test rows  = 2417, videos = 22
total rows = 11902
video leakage = 0
```

Action order:

```text
0 = maintain_speed
1 = reduce_speed
2 = stop_car
```

Each sample has:

```text
input_frames: 15 observed frame indices
target_frame: start + 15
action_soft_target: annotator vote distribution over 3 actions
action_name: majority action
paper_effective_weight / sample_weight
raw explanation_text
raw reasoning_text
```

Formal model input is the 15 observed frames only:

```text
input_frames = [start, ..., start+14]
target_frame = start+15 is label alignment and optional DAMO-image baseline input only.
```

Do not feed `target_frame` image to ACPR-InteractFlow++ formal model.

### 4.4 Exp29 boundary

Exp29 labels are reconstructed text clusters, not official human-defined PSI classes.

Use them as:

```text
weak explanation labels
cluster nodes with medoid/top phrases/reliability
auxiliary explanation supervision
```

Do not treat all-zero Exp29 rows as 29 true negatives. All-zero can mean missing text or no reliable cluster.

---

## 5. DAMO-compatible metrics

Formal evaluation must reproduce the local DAMO PSI evaluator on its own prediction format before any model training.

### 5.1 Action metrics

At minimum report:

```text
Act_oAcc
Act_mAcc
Act_macro_F1
Act_weighted_F1
Maintain recall / precision / F1
Reduce recall / precision / F1
Stop recall / precision / F1
soft_target_KL
ECE
confusion_matrix
prediction_rate
```

`Act_mAcc` and per-class recall are critical because Stop is a minority but safety-sensitive class.

### 5.2 Explanation metrics

For 29 weak explanation labels report:

```text
Exp_oF1
Exp_mF1
Exp_mAP
Exp_micro_F1
Exp_macro_F1
per-label F1/AP
positive-mask-only metrics
all-zero-row count
```

Rows with no explanation text or no positive Exp29 mask must not dominate Exp29 supervision metrics.

### 5.3 Joint metrics and best selection

Formal epoch bests:

```text
checkpoint_latest.pth
checkpoint_best_action.pth
checkpoint_best_exp.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
```

Default test-selected joint score:

\[
joint = 0.60 \cdot Act\_mAcc + 0.25 \cdot StopF1 + 0.15 \cdot Exp\_mF1
\]

Also write the DAMO-style joint score used by the local DAMO reproduction if different.

Best-test selector:

1. maximize the configured `joint`;
2. tie-break by `Act_mAcc`;
3. tie-break by `StopF1`;
4. tie-break by `Exp_mF1`;
5. tie-break by lower action soft KL.

No validation selection in the formal run.

---

## 6. Formal namespace and import isolation

Add:

```text
fate_oia/acpr_interactflow/
```

Formal entrypoints:

```text
python -m fate_oia.engine.train_acpr_interactflow_psi
python -m fate_oia.engine.eval_acpr_interactflow_psi
python -m fate_oia.engine.run_acpr_interactflow_preflight
python -m fate_oia.engine.audit_acpr_interactflow
python -m fate_oia.engine.profile_acpr_interactflow
python -m fate_oia.engine.export_acpr_interactflow_visuals
python -m fate_oia.engine.supervise_acpr_interactflow_foreground
```

Formal import graph must not instantiate:

```text
ACPROIAModel as the formal PSI model
train_acpr_oia as formal trainer
BDD-OIA dataset classes as formal PSI dataset
old cached-logit/tail-adapter paths
FATE-X / BDD-X trainers
```

It may reuse low-level modules listed in Section 1.

---

## 7. Typed tensor/output contracts

Create `fate_oia/acpr_interactflow/types.py`.

### 7.1 Batch

```python
@dataclass
class PSIInteractFlowBatch:
    frames: Tensor                     # [B,15,3,H,W]
    action_soft_target: Tensor         # [B,3]
    action_hard: Tensor                # [B]
    exp29_target: Tensor               # [B,29]
    exp29_mask: Tensor                 # [B,29] or [B,1]
    sample_weight: Tensor              # [B]
    paper_effective_weight: Tensor     # [B]
    input_frame_indices: Tensor        # [B,15]
    target_frame_index: Tensor         # [B]
    video_ids: list[str]
    sample_ids: list[str]
    raw_explanation_text: list[str]
    raw_reasoning_text: list[str]
    frame_paths: list[list[str]]
```

### 7.2 Visual output

```python
@dataclass
class InteractVisualOutput:
    anchor_indices: tuple[int, ...]     # default (0,3,6,9,12,14) or full15 if configured
    anchor_features: Tensor             # [B,A,N,D]
    anchor_hw: tuple[int, int]
    motion_tokens: Tensor               # [B,15,Dm]
    lowres_motion_maps: Tensor          # [B,15,Hm,Wm,Dm]
    selected_layer_tokens: dict[str, Tensor]
```

### 7.3 Predicate field

```python
@dataclass
class InteractPredicateField:
    predicate_names: tuple[str, ...]      # OIA 32 + PSI-specific predicates
    oia_predicate_names: tuple[str, ...]  # exact 32
    psi_predicate_names: tuple[str, ...]
    logits: Tensor                        # [B,15,P]
    probabilities: Tensor                 # [B,15,P]
    tokens: Tensor                        # [B,15,P,D]
    evidence_maps: Tensor                 # [B,A,P,H,W] for anchor frames
    evidence_kind: Tensor                 # observed_anchor/interpolated
    confidence: Tensor                    # [B,15,P]
    centroid: Tensor                      # [B,15,P,2]
    relative_motion: Tensor               # [B,14,P,2]
    corridor_mass: Tensor                 # [B,15,P,4]
    temporal_stats: dict[str, Tensor]
    transfer_gate: Tensor                 # [P]
```

### 7.4 Interaction flow

```python
@dataclass
class InteractionFlowState:
    regime_names: tuple[str, ...]
    phase_names: tuple[str, ...]
    source_names: tuple[str, ...]
    corridor_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    factor_tokens: Tensor                 # [B,15,F,D]
    factor_logits: Tensor                 # [B,15,F]
    factor_probs: Tensor                  # [B,15,F]
    factor_to_predicate: Tensor           # [B,15,F,P]
    factor_to_corridor: Tensor            # [B,15,F,4]
    lag_weights: Tensor                   # [B,F,5]
    lag_aligned_tokens: Tensor            # [B,F,D]
    evidence_maps: Tensor                 # [B,A,F,H,W]
    lineage: list[dict]
```

### 7.5 Decision ledger

```python
@dataclass
class InteractionDecisionLedger:
    action_names: tuple[str, str, str]     # maintain, reduce, stop
    global_logits: Tensor                  # [B,3]
    raw_state_contributions: Tensor        # [B,F,3]
    benefit_gate: Tensor                   # [B,F,1] or [B,F,3]
    gated_state_contributions: Tensor      # [B,F,3]
    final_logits: Tensor                   # [B,3]
    final_probs: Tensor                    # [B,3]
    contribution_attention: Tensor         # [B,F]
    benefit_target: Tensor | None
```

Audit identity:

\[
final\_logits = global\_logits + gated\_state\_contributions.sum(dim=1)
\]

### 7.6 Explanation output

```python
@dataclass
class Exp29Output:
    logits: Tensor                         # [B,29]
    probabilities: Tensor                  # [B,29]
    cluster_attention_to_factors: Tensor   # [B,29,F]
    cluster_reliability: Tensor            # [29]
    medoid_text: list[str]
```

### 7.7 Formal output

```python
@dataclass
class ACPRInteractFlowPPOutput:
    total_loss: Tensor
    loss_components: dict[str, Tensor]
    visual: InteractVisualOutput
    predicates: InteractPredicateField
    interaction_flow: InteractionFlowState
    ledger: InteractionDecisionLedger
    exp29: Exp29Output
    diagnostics: dict[str, Any]
```

No positional tuple parsing.

---

## 8. Data loader implementation

Files:

```text
fate_oia/acpr_interactflow/psi_damo_dataset.py
fate_oia/engine/acpr_interactflow_data.py
```

Requirements:

- read pkl/jsonl samples and Exp29 pkl by split;
- verify row count and sample ID alignment;
- verify video-level split leakage is zero;
- load exactly 15 observed frames by `input_frames`;
- never load target frame for formal model input;
- preserve raw text;
- expose `paper_effective_weight`;
- support aspect-ratio preserving transform, default `320×576` or `288×512`;
- support a separate DAMO target-frame image baseline loader for metric parity and baseline reproduction only.

Frame path resolution must be explicit and audited. If frame path is missing, fail loudly with sample ID, video ID, requested frame, and searched paths.

---

## 9. Visual encoder

Files:

```text
fate_oia/acpr_interactflow/visual_encoder.py
fate_oia/acpr_interactflow/motion_path.py
```

### 9.1 Default formal encoder

Use a hybrid evidence-efficient design:

```text
High-resolution DINO ViT-S/8 evidence path:
  anchor frames = [0,3,6,9,12,14]
  resolution = 320×576 or 288×512
  DINO backbone frozen by default
  last-N-layer token fusion
  optional LoRA/adapter on selected final blocks

Fast 15-frame motion path:
  all 15 frames at lower spatial resolution
  lightweight conv/TSM-style temporal exchange
  output motion tokens and low-resolution motion maps
```

The model still trains directly from images. This is not a feature cache.

### 9.2 Optional full15 DINO mode

Implement `dino_all15` mode for ablation/upper bound, but do not make it the default unless the throughput gate proves it is stable and faster or more accurate on the mechanism subset.

### 9.3 Why this is the formal default

PSI scenes need high-resolution evidence for pedestrians/crosswalk/occlusion, but the label is a short 15-frame interaction decision. Full 15-frame high-res DINO on every frame is usually unnecessary. The formal design concentrates heavy spatial evidence on anchor frames and tracks temporal changes at the predicate level.

### 9.4 Precision

Use native BF16 autocast on CUDA. Frozen DINO can safely run in no-grad BF16 unless a precision audit shows feature instability. No forced FP32 path unless explicitly audited.

---

## 10. Predicate ontology

Files:

```text
configs/acpr_interactflow_predicates.yaml
fate_oia/acpr_interactflow/predicate_ontology.py
```

Total predicates:

```text
OIA base predicates: exact 32 from configs/acpr_scene_predicates.yaml
PSI interaction predicates: default 16
total P = 48
```

### 10.1 OIA 32

Copy names/order exactly from `configs/acpr_scene_predicates.yaml`.

### 10.2 PSI-specific predicates

Default:

```text
pedestrian_waiting
pedestrian_approaching_ego_path
pedestrian_entering_ego_lane
pedestrian_crossing_ego_path
pedestrian_moving_away
pedestrian_group
pedestrian_on_curb
pedestrian_in_roadway
pedestrian_looking_towards_ego
crosswalk_conflict
side_occlusion_risk
ego_path_clear
vehicle_yielding_ahead
vehicle_passing_pedestrian
traffic_signal_constraint
intersection_constraint
```

Each predicate has:

```text
name
group
region/corridor prior
positive phrases
contradiction phrases
structural weak target rules
supervision mode: text_nnpu | structural_weak | consistency_only
```

Do not fabricate strong text labels for predicates not reliably mentioned.

---

## 11. OIA-to-PSI predicate transfer

File:

```text
fate_oia/acpr_interactflow/predicate_transfer.py
```

Load the ACPR-CalAlign source checkpoint from the source branch/run record.

For each OIA predicate:

\[
q_k^{PSI} =
W_o q_k^{OIA} +
W_n E_{\text{text}}(name_k) +
r_k
\]

For PSI-specific predicates:

\[
q_k^{PSI} =
W_n E_{\text{text}}(name_k) +
r_k
\]

where:

- \(q_k^{OIA}\) is the learned ACPR predicate query/prototype;
- \(E_{\text{text}}\) is a true frozen text encoder embedding from local BERT/Sentence-BERT model;
- \(r_k\) is trainable PSI residual;
- optional `transfer_gate` controls OIA prior strength.

Do not use hash-byte pseudo-embeddings.

Write source checkpoint path, SHA256, source tensor key, source shape, mapped shape, and loaded predicate names.

Formal review fails if OIA checkpoint resolution is ambiguous or the exact 32 names cannot be verified.

---

## 12. Dynamic predicate field

File:

```text
fate_oia/acpr_interactflow/dynamic_predicate_field.py
```

For anchor frame \(a\):

\[
A_{a,k} =
entmax_{1.5}(q_{a,k}^{T}K_a + b_k^{region})
\]

\[
e_{a,k} = \sum_{x,y} A_{a,k}(x,y)V_a(x,y)
\]

For all frames \(t=0..14\):

```text
use observed anchor evidence where available
use motion path and recurrent state to update non-anchor proxy evidence
```

Temporal update:

\[
h_{t,k} =
GRU(h_{t-1,k}, [e_{t,k}, p_{t,k}, \Delta\mu_{t,k}, m_{t,k}^{corridor}])
\]

Also compute TCN statistics:

```text
trend
velocity
acceleration
volatility
presence_rate
visibility
```

Confidence combines:

```text
presence probability
entmax concentration
evidence feature agreement
temporal consistency
visibility/border penalty
text/structural calibration reliability
```

Outputs include `evidence_kind` so visualization distinguishes observed anchor evidence from interpolated temporal proxy.

---

## 13. nnPU-CalAlign for predicates and interaction states

Files:

```text
fate_oia/acpr_interactflow/nnpu_calalign.py
configs/acpr_interactflow_text_rules.yaml
```

### 13.1 Labels

From `explanation_text + reasoning_text`:

```text
explicit support       → positive
explicit contradiction → reliable negative
otherwise              → unlabeled
```

For structural predicates:

```text
use weak structural evidence and consistency, not fake text positives
```

### 13.2 Nonnegative PU risk

\[
R_k^{PU}
=
\pi_k R_k^+
+
\max(0, R_k^{u-} - \pi_k R_k^{+-})
\]

No hard negative BCE for unknown/unmentioned samples.

### 13.3 Train-only calibration

Maintain train-only histograms for known positives and reliable negatives.

At epoch end update:

```text
class prior
temperature
positive threshold
negative threshold
known-label precision/recall estimate
```

Test evaluation cannot update priors/thresholds/histograms.

---

## 14. Interaction-flow state composer

Files:

```text
fate_oia/acpr_interactflow/interaction_flow.py
fate_oia/acpr_interactflow/interaction_grammar.py
```

### 14.1 States

Regime:

```text
clear_to_go
caution_required
yielding_required
stop_required
```

Phase:

```text
waiting
approaching
entering
crossing
leaving
uncertain
```

Source:

```text
pedestrian_conflict
crosswalk_context
traffic_signal
front_vehicle_yielding
side_occlusion
intersection_constraint
```

Corridor:

```text
left_sidewalk_zone
center_ego_path
right_sidewalk_zone
crosswalk_zone
```

Factor set defaults to regime + phase + source = 16 decision factors. Corridors are support tokens, not final decision factors unless configured.

### 14.2 Construction

Each factor query sparsely attends with entmax over:

```text
dynamic predicate tokens
predicate temporal statistics
corridor tokens
motion tokens
semantic-name embeddings
grammar support/contradiction priors
```

Store support lineage:

```text
top predicates
top corridors
pattern/phase evidence
anchor evidence map
confidence
```

### 14.3 Weak semantic targets

Build detached weak targets from predicate states:

```text
pedestrian entering/crossing + ego path conflict → stop_required / pedestrian_conflict
pedestrian waiting + path clear → clear_to_go
approaching + occlusion → caution_required / uncertain
traffic light red/green → traffic_signal constraint
front vehicle yielding → front_vehicle_yielding
```

These are weak regularizers, not hard labels.

---

## 15. Response-lag alignment

File:

```text
fate_oia/acpr_interactflow/response_lag.py
```

For each factor \(k\), use lags 0..4 frames:

\[
\lambda_{k,l} = softmax_l(q^{decision} W_l s_{15-l,k})
\]

\[
\tilde{s}_k = \sum_{l=0}^{4} \lambda_{k,l}s_{15-l,k}
\]

Requirements:

- causal: only observed frames;
- out-of-range masked;
- lag weights sum to one;
- lag-disabled mode for audit;
- synthetic delayed-event test recovers known lag;
- real temporal reverse changes phase/lag decisions.

---

## 16. Benefit-gated exact decision ledger

File:

```text
fate_oia/acpr_interactflow/decision_ledger.py
```

### 16.1 Global logits

A standard global visual decision branch predicts:

```text
z_global [B,3]
```

from temporal visual/predicate summary. It must be independently supervised.

### 16.2 State contributions

For each factor \(k\):

\[
\Delta z_k = a_k d_k
\]

where \(\Delta z_k \in \mathbb{R}^3\) contributes to maintain/reduce/stop logits.

### 16.3 Benefit gate

\[
z^{candidate}=z^{global}+\sum_k\Delta z_k
\]

\[
benefit = KL(y,z^{global}) - KL(y,z^{candidate})
\]

\[
\alpha_k = sigmoid(g(s_k,z^{global},confidence_k))
\]

Use detached benefit signal to train gates. The gate can close if the interaction branch would hurt action prediction.

### 16.4 Final logits

\[
z^{final}=z^{global}+\sum_k\alpha_k\Delta z_k
\]

This exact identity is mandatory and audited.

### 16.5 Non-degradation hinge

\[
L_{safe}=ReLU(KL(y,z^{final}) - stopgrad(KL(y,z^{global})) + margin)
\]

This prevents the interaction branch from harming the main decision.

Do not use `abs(contribution)` as the residual loss.

---

## 17. Exp29 weak explanation head

Files:

```text
fate_oia/acpr_interactflow/exp29_head.py
fate_oia/acpr_interactflow/cluster_semantics.py
```

### 17.1 Cluster metadata

Read label embedding file:

```text
label_embedding/label_embedding_psi_exp29.pkl
label_embedding/label_embedding_psi_exp29.json
```

Create cluster metadata:

```text
cluster id
medoid text
top phrases
embedding
action prior
state support prior
reliability score
```

### 17.2 Prediction

Exp29 head reads:

```text
interaction-flow factor tokens
exact decision contributions
predicate support tokens
global decision hidden
```

Output:

```text
exp29 logits [B,29]
factor attention [B,29,F]
```

### 17.3 Loss

Use masked reliability-weighted ASL/BCE:

```text
positive clusters supervised
unknown all-zero rows not treated as 29 hard negatives
reliability weights downweight noisy clusters
```

### 17.4 Contribution alignment

Let:

\[
q_k = Normalize(\sum_c |\Delta z_{k,c}|)
\]

Align explanation factor attention with real contribution:

\[
L_{align}=JS(A_{exp\rightarrow factor}, q)
\]

---

## 18. Losses

File:

```text
fate_oia/losses/acpr_interactflow_losses.py
```

Configured default weights:

```text
action_final_soft_kl          1.00
action_global_soft_kl         0.50
ledger_residual_soft_kl       0.20
non_degradation_hinge         0.10

exp29_masked_asl              0.30
predicate_nnpu                0.10
interaction_state_semantic    0.05
response_lag_consistency      0.03
contribution_alignment_js     0.05
temporal_consistency          0.02
gate_entropy                  0.002
group_sparsity                0.002
```

Forbidden formal losses:

```text
all-zero Exp29 as all-negative BCE
contribution magnitude as residual objective
action hard CE as sole action supervision
state probability mean minimization
pattern logit square-to-zero
HardPair imported from BDD-OIA as formal PSI objective
test-threshold leakage
cached-logit residual adapter
```

Every nonzero loss writes:

```text
raw value
weight
weighted value
finite status
gradient target modules
```

---

## 19. Training protocol

File:

```text
fate_oia/engine/train_acpr_interactflow_psi.py
```

### 19.1 Formal run

```text
epochs = 30
precision = BF16 autocast when CUDA supports it
optimizer = AdamW
scheduler = 5% warmup + cosine decay
gradient clip = 1.0
direct image training
feature_cache = false
token_cache = false
metric early stop = false
```

### 19.2 Fixed trainability

Default:

```text
DINO backbone frozen
DINO adapter/LoRA trainable if enabled
OIA transferred base query prior detached
OIA query mapper/gate/residual trainable
dynamic predicate GRU/TCN trainable
motion path trainable
interaction state reasoner trainable
decision ledger trainable
Exp29 head trainable
```

No staged freezing. Warm-up of contribution gate is allowed.

### 19.3 Optimizer groups

Use explicit groups:

```text
dino_adapter                   1.0e-5
predicate_transfer             1.0e-4
dynamic_predicate              1.0e-4
motion_path                    1.0e-4
interaction_flow               1.0e-4
response_lag                   7.5e-5
decision_ledger                1.0e-4
exp29_head                     1.0e-4
calibration_thresholds         5.0e-5
```

A single global LR is forbidden.

### 19.4 Checkpoints

Save atomically:

```text
checkpoint_latest.pth before every evaluation
checkpoint_best_action.pth
checkpoint_best_exp.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
```

Reject `.tmp` checkpoint resume.

### 19.5 Evaluation

User requested:

```text
best selected using test
after every epoch evaluate test only
```

No validation loader in formal training. The val split can be loaded in nonformal data-audit mode only.

---

## 20. Memory and throughput

File:

```text
fate_oia/engine/profile_acpr_interactflow.py
```

Target GPU: RTX 5880, 48 GiB.

Preferred actual peak reserved:

```text
34–42 GiB
```

Hard cap:

```text
44 GiB
```

Do not allocate dummy tensors.

Probe candidates:

```text
micro batch 16 / accumulation 2
micro batch 12 / accumulation 3
micro batch 8  / accumulation 4
micro batch 6  / accumulation 5
micro batch 4  / accumulation 8
micro batch 3  / accumulation 11
micro batch 2  / accumulation 16
```

For each:

```text
10 warm-up batches
100 measured full forward/backward batches
direct frames
all formal losses
real dataloader
BF16 if available
```

Record:

```text
data_time
visual_time
predicate_time
interaction_time
decision_time
exp_time
backward_time
optimizer_time
samples/sec
peak allocated/reserved
projected train epoch time
projected full test eval time
```

Selection:

```text
highest stable samples/sec under hard cap
not necessarily highest memory use
```

Formal training blocked if:

```text
projected train epoch + full test eval > 2.5 hours
data_time fraction > 25%
NaN/Inf/skip observed
PU positive count = 0 in measured window
state contribution = 0 across measured window
```

---

## 21. Evaluation and traffic-flow influence verification

Files:

```text
fate_oia/engine/eval_acpr_interactflow_psi.py
fate_oia/explain/acpr_interactflow_faithfulness.py
```

### 21.1 Per-epoch standard evaluation

Full test:

```text
DAMO-compatible action metrics
DAMO-compatible Exp29 metrics
decision ledger stats
fixed 256-sample lightweight influence audit
```

### 21.2 Lightweight audit every epoch

On fixed 256 test samples:

```text
full
global-only / all interaction states off
top-state-off
temporal-reverse
lag-disabled
```

This audit is not used for best selection.

### 21.3 Full best-checkpoint audit

On `checkpoint_best_test.pth` and full test:

```text
global-only
regime-off
phase-off
source-off
individual factor-off
predicate-off
evidence-tube-off
equal-mass random evidence deletion, 5 seeds
temporal reverse
temporal shuffle
lag disabled
last-frame-only
prefix 5/10/15 frames
```

Interventions must rerun from the earliest affected layer:

```text
evidence-off → predicate field onward
predicate-off → interaction flow onward
factor-off → response lag and ledger onward
temporal reverse/shuffle → visual/motion/temporal/predicate onward
```

Changing only a display tensor is forbidden.

### 21.4 Influence metrics

Decision Dependence:

```text
ΔKL = KL(full) - KL(state-off)
ΔAct_mAcc
ΔStopF1
```

Evidence Specificity:

\[
ES = \Delta Error_{evidence} - E[\Delta Error_{random}]
\]

Temporal Necessity:

\[
TN = Error_{reverse/shuffle} - Error_{full}
\]

Lag Necessity:

\[
LN = Error_{lag0} - Error_{learnedlag}
\]

Contribution Fidelity:

```text
Spearman correlation between exact contribution ranking and factor-off effect ranking
```

Explanation Consistency:

```text
JS / rank correlation between Exp29 factor attention and exact decision contributions
```

Use the phrase:

```text
model-level counterfactual dependence
```

Do not claim real-world causality.

---

## 22. Visualization

Files:

```text
fate_oia/explain/acpr_interactflow_renderer.py
fate_oia/explain/acpr_interactflow_atlas.py
```

Generate **Dynamic Interaction Decision Ledger**.

Per-case PNG + source JSON:

1. 15-frame strip with selected keyframes;
2. predicate evidence tubes for pedestrian/crosswalk/traffic light/ego path/occlusion;
3. interaction-flow ribbons:
   ```text
   clear_to_go
   caution_required
   yielding_required
   stop_required
   pedestrian_conflict
   occluded_risk
   crosswalk_context
   ```
4. response-lag panel:
   ```text
   when state peaks, which lag is selected, how it affects target decision
   ```
5. exact decision waterfall:
   ```text
   global logits
   each state contribution
   benefit gate
   final logits/probabilities
   ```
6. Exp29 contribution-alignment panel:
   ```text
   predicted clusters
   medoid text
   factor attention
   exact factor contribution
   ```
7. counterfactual twin:
   ```text
   full
   state-off
   evidence-off
   equal-mass random
   temporal reverse
   ```
8. tensor lineage, sample ID, video ID, frame indices, checkpoint SHA, config SHA.

Dataset atlas:

```text
state→action matrix
state→StopF1 effect
predicate→state support
response-lag distributions
evidence-off vs random distributions
global-only vs full performance
success/failure prototypes
per-class failure cases
```

No manual boxes, no fabricated values, no placeholder HTML.

---

## 23. Required files to add

```text
configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml
configs/acpr_interactflow_predicates.yaml
configs/acpr_interactflow_text_rules.yaml
configs/acpr_interactflow_state_grammar.yaml

fate_oia/acpr_interactflow/__init__.py
fate_oia/acpr_interactflow/config.py
fate_oia/acpr_interactflow/types.py
fate_oia/acpr_interactflow/psi_damo_dataset.py
fate_oia/acpr_interactflow/psi_frame_resolver.py
fate_oia/acpr_interactflow/psi_metrics.py
fate_oia/acpr_interactflow/visual_encoder.py
fate_oia/acpr_interactflow/motion_path.py
fate_oia/acpr_interactflow/predicate_ontology.py
fate_oia/acpr_interactflow/predicate_transfer.py
fate_oia/acpr_interactflow/dynamic_predicate_field.py
fate_oia/acpr_interactflow/nnpu_calalign.py
fate_oia/acpr_interactflow/interaction_flow.py
fate_oia/acpr_interactflow/interaction_grammar.py
fate_oia/acpr_interactflow/response_lag.py
fate_oia/acpr_interactflow/decision_ledger.py
fate_oia/acpr_interactflow/cluster_semantics.py
fate_oia/acpr_interactflow/exp29_head.py
fate_oia/acpr_interactflow/interventions.py
fate_oia/acpr_interactflow/model.py

fate_oia/losses/acpr_interactflow_losses.py

fate_oia/engine/train_acpr_interactflow_psi.py
fate_oia/engine/eval_acpr_interactflow_psi.py
fate_oia/engine/profile_acpr_interactflow.py
fate_oia/engine/run_acpr_interactflow_preflight.py
fate_oia/engine/audit_acpr_interactflow.py
fate_oia/engine/export_acpr_interactflow_visuals.py
fate_oia/engine/build_acpr_interactflow_atlas.py
fate_oia/engine/supervise_acpr_interactflow_foreground.py

fate_oia/explain/acpr_interactflow_renderer.py
fate_oia/explain/acpr_interactflow_atlas.py
fate_oia/explain/acpr_interactflow_faithfulness.py

scripts/FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1

docs/runbooks/ACPR_InteractFlowPP_V1_Implementation_Plan.md
docs/runbooks/ACPR_InteractFlowPP_V1_Implementation_Manifest.json
.codex/skills/acpr-interactflowpp-implementation-audit/SKILL.md
```

---

## 24. Existing files that may be modified

Modify only when necessary and backward-compatibly:

```text
fate_oia/models/acpr_dino_field.py
fate_oia/models/acpr_scene_predicate_head.py
fate_oia/transforms.py
fate_oia/losses/__init__.py
.gitignore
```

Do not make `train_acpr_oia.py` the formal PSI trainer.

---

## 25. Required tests

Create:

```text
tests/acpr_interactflow/
```

### Dataset/protocol

```text
test_psi_damo_dataset_counts.py
test_psi_damo_sample_alignment.py
test_video_leakage_zero.py
test_formal_input_uses_input_frames_not_target_frame.py
test_exp29_all_zero_not_all_negative.py
test_damo_metric_parity.py
```

### Visual/predicate

```text
test_oia_32_predicate_transfer.py
test_psi_predicate_ontology.py
test_true_text_name_embedding.py
test_dynamic_predicate_shapes.py
test_entmax_evidence.py
test_temporal_order_changes_predicates.py
test_anchor_evidence_kind.py
```

### PU/state/lag

```text
test_nnpu_nonnegative_risk.py
test_unknown_not_negative.py
test_calalign_train_only.py
test_interaction_state_grammar.py
test_response_lag_synthetic.py
test_temporal_reverse_changes_phase.py
```

### Decision/explanation

```text
test_exact_decision_ledger.py
test_benefit_gate_target_direction.py
test_non_degradation_hinge_direction.py
test_soft_action_kl_uses_soft_target.py
test_exp29_masked_asl.py
test_contribution_alignment.py
```

### Train/eval/audit

```text
test_optimizer_groups.py
test_bf16_runtime.py
test_atomic_checkpoint.py
test_test_only_best_selection.py
test_profile_is_real.py
test_intervention_recompute.py
test_equal_mass_random.py
test_visual_canvas_schema.py
test_atlas_schema.py
test_foreground_supervisor.py
test_review_pass_binding.py
test_real_direct_image_smoke.py
```

---

## 26. Blocking pre-training gates

Formal training is forbidden until every gate passes.

### Gate A — Git/worktree/config/import graph

- target worktree exists;
- branch `acpr_interactflow_pp_v1`;
- clean;
- local SHA equals GitHub branch SHA;
- source SHA recorded;
- formal import graph isolated;
- every config field has runtime/audit consumer.

### Gate B — Dataset and DAMO metric parity

- exact 8873/612/2417 counts;
- Exp29 dim 29;
- action order maintain/reduce/stop;
- no video leakage;
- metric bridge reproduces local DAMO evaluator on saved DAMO predictions or synthetic exact fixtures.

### Gate C — OIA transfer

- exact OIA 32 names loaded;
- source checkpoint SHA recorded;
- query/prototype key loaded;
- true text embeddings used;
- gradients reach transfer mapper/residual.

### Gate D — Real direct-image smoke

- 8 train samples;
- 8 test samples;
- at least 8 optimizer steps;
- frames shape `[B,15,3,H,W]`;
- no feature/token cache;
- test eval after smoke;
- checkpoint_latest exists.

### Gate E — Gradient chain

Finite nonzero gradients for:

```text
predicate transfer residual
dynamic predicate temporal module
motion path
interaction state reasoner
response lag
decision ledger
exp29 head
```

Frozen DINO base has no gradients unless adapter mode intentionally unfreezes it.

### Gate F — 128-sample mechanism fit

On deterministic 128 train samples, bounded updates must improve:

```text
global action KL
final action KL
ledger residual objective
predicate known-label PU risk
interaction state weak loss
exp29 positive-mask loss
contribution alignment
```

Reject collapse:

```text
all maintain predictions
all stop impossible
all predicates constant
all states constant
all benefit gates zero
all state contributions zero
exp29 attention all uniform
```

### Gate G — Temporal/lag necessity

Synthetic and real subset tests:

```text
reverse changes approaching/entering/crossing phase
lag disabled changes decision on delayed cases
last-frame-only differs from full15
```

### Gate H — Intervention

On real test samples:

```text
state-off changes action probabilities
evidence-off changes predicates/states/actions
equal-mass random matched
factor contribution ranking correlates with factor-off effect in smoke subset
```

### Gate I — Visualization

One real complete Dynamic Interaction Decision Ledger PNG/JSON and mini Atlas.

### Gate J — Throughput/memory

100 measured real steps; projected train+test epoch under configured cap; memory stable; no dummy allocation.

### Gate K — Independent review pass

Only Agent B may write:

```text
.background_runs/acpr_interactflow_pp_v1_preflight/REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt
```

---

## 27. Review pass

The review pass binds:

```text
source branch/SHA
target branch/SHA
clean status
GitHub remote SHA
plan/config/skill/manifest hashes
all gate reports
selected batch config
throughput projection
```

Any source/config/script/test change invalidates the pass.

---

## 28. Formal training launch

After review pass:

```powershell
Set-Location E:\sbw\FATE_Drive\fate_oia_acpr_interactflow_pp_worktree

powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1 `
  -Config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  -RequireReviewPass
```

Supervisor must be foreground attached.

Forbidden:

```text
Start-Process
Start-Job
nohup
shell &
schtasks
hidden window
metric early stop
```

On recoverable failure:

```text
preserve log/checkpoint
systematic debugging
add regression test
fix
commit/push
invalidate review pass
rerun audit
resume latest
continue
```

---

## 29. Required epoch artifacts

Each epoch:

```text
epoch_XXX/
  action_metrics.json
  exp29_metrics.json
  joint_metrics.json
  loss_components.jsonl
  gradient_norms.json
  predicate_stats.json
  nnpu_calibration.json
  interaction_state_stats.json
  response_lag_stats.json
  decision_ledger_stats.json
  lightweight_interaction_influence.json
  predictions_action.jsonl
  predictions_exp29.jsonl
  fixed_case_intermediate_outputs.jsonl
```

Run root:

```text
config_resolved.yaml
run_manifest.json
git_provenance.json
psi_dataset_contract.json
damo_metric_parity.json
oia_transfer_report.json
optimizer_groups.json
throughput_profile.json
checkpoint_latest.pth
checkpoint_best_action.pth
checkpoint_best_exp.pth
checkpoint_best_joint.pth
checkpoint_best_test.pth
train.log
supervisor_live_status.json
supervisor_decisions.jsonl
run_complete.json
```

No fake zeros. Missing values require explicit unavailable reason.

---

## 30. Definition of implementation complete

Implementation complete only when:

- formal namespace exists;
- PSI dataset protocol is verified;
- DAMO metric parity is proven;
- exact OIA 32 transfer works;
- dynamic PSI-specific predicates execute;
- nnPU has real known positives/reliable negatives;
- Exp29 all-zero rows are not treated as all negative;
- interaction-flow states are computed from dynamic predicates;
- response lag affects decision;
- exact ledger identity holds;
- benefit gate direction is correct;
- contribution-aligned Exp29 head is active;
- all losses have correct sign and gradients;
- interventions rerun downstream computations;
- visualization is tensor-linked and non-placeholder;
- throughput/memory profile is real;
- independent audit pass binds exact clean pushed SHA.

File existence is insufficient.

---

## 31. Definition of experiment complete

Formal experiment complete only after:

- all 30 epochs finish;
- full test evaluation after every epoch;
- all best/latest checkpoints saved;
- best-test full influence audit finishes;
- Dynamic Interaction Decision Ledger cases exported;
- Atlas generated;
- canonical three records updated;
- local/GitHub SHA equality verified;
- `run_complete.json` written.

---

## 32. Hard failures

Do not authorize formal training if any is true:

```text
formal model imports ACPROIAModel as main PSI model
target_frame image enters formal input
Exp29 all-zero treated as 29 negatives
action soft targets converted to hard labels as sole supervision
OIA checkpoint unresolved
OIA names/order mismatch
hash pseudo-embeddings used for predicate names
predicate path has zero gradients
state path disconnected from decision
lag configured but unused
ledger does not exactly reconstruct final logits
benefit gate always zero or has no target
contribution magnitude minimized as residual
intervention changes only display tensors
DAMO metric parity missing
profile report fabricated
review pass stale
worktree dirty
local/GitHub SHA mismatch
```
