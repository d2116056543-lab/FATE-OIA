# TIDA Trajectory-Relational Traffic Credit V5 Design

## Objective

Increase the strong frozen-image TIDA action result from Act_mF1 0.7839 toward 0.79 on the fixed 885-clip test split by learning action-relevant traffic trajectories, without changing the image backbone, using reason evidence in action, fitting parameters on test, or hiding a threshold-only gain. Preserve the existing Exp branch exactly.

## Root Cause

V4.1 independently selects top-12 patches in every frame and soft-matches only adjacent sets. It then averages all matched points into one displacement per action. This loses point identity, trajectory shape, occlusion state, and interactions between moving participants. The resulting residual has a positive mean signed margin but insufficient ranking magnitude, and small train-calib threshold changes amplify boundary errors.

## Architecture

### Cycle-Consistent Action Trajectories

Use the final history frame's action-attentive patches as stable anchors. Propagate each anchor backward through normalized DINO patch features with bidirectional soft correspondence. A match is reliable only when forward and backward assignments agree. Return per-action trajectories `[B,A,K,T,2]`, aligned appearance `[B,A,K,T,D]`, visibility/confidence, and cycle error. No external optical-flow or tracking model is used.

### Relational Trajectory Tokens

Encode each trajectory with appearance, displacement, velocity, acceleration, radial expansion, confidence, and an eight-bin soft histogram of displacement directions. A temporal encoder models each trajectory; a relational encoder models differences among trajectories. Estimate common camera/static motion robustly across trajectories and expose both common and exclusive motion. Each action query cross-attends the K trajectory tokens and produces a bounded zero-initialized action-only residual.

### Action Utility And Calibration Stability

Train the raw traffic residual with final-action ASL, boundary correction, cross-sample ranking, cycle consistency, and same-video selected-vs-control trajectory deletion. The correction loss is action-specific and emphasizes train examples near the frozen base decision boundary. A per-action trust gate starts at zero and can only increase when train-core signed utility is positive. Formal deploy uses thresholds fit on train-calib, with a stability tie-break to the frozen strong thresholds; test labels never update thresholds or gates.

### Firewalls

- `action_logits_final = semantic_action_logits + bounded_traffic_delta`.
- Reason logits and reason parameters cannot receive traffic gradients or deltas.
- Action-set, graph, reason logits, test labels, and BDD100K geometry do not enter the traffic forward path.
- V5 disabled or zero-initialized is exactly V4.1/strong-CTU compatible.

## Evaluation

Primary metrics remain Act_mF1, Act_oF1, and Act_mAP on all 885 test clips. Traffic-specific evidence includes:

- Dynamic-conditional Act_mF1/mAP over train-derived motion quantiles.
- Prefix anticipation AUC and earliest-correct-frame gain.
- Corrective-to-harmful flip ratio at train-calib deploy thresholds.
- GT-signed margin and benefit rate by action.
- Selected-trajectory deletion minus matched-random deletion with bootstrap confidence interval.
- Original versus reverse, shuffled, and repeated-last temporal interventions.
- Track cycle consistency, visibility coverage, trajectory diversity, and common/exclusive motion ratio.
- Per-case trajectory overlays, direction histograms, action contribution waterfalls, and prefix confidence curves.

## Training Decision

Run a one-checkpoint focused pilot from the strong EMA checkpoint. Stage 1 trains trajectory representation and raw residual while deployed trust is zero. Stage 2 opens per-action trust using train-core utility and keeps train-calib thresholds stable. A formal long run is allowed only if the independent test shows no action degradation and either Act_mF1 improves materially or Act_mAP/dynamic/anticipation evidence shows a credible trend. A gate pass with weak absolute metrics is not sufficient.

## Research Basis

The design adapts, rather than copies, four findings: temporally-aware DINO features improve correspondence (Chrono, CVPR 2025); semantic point sampling and intra/inter-trajectory motion features improve action recognition (Trokens, ICCV 2025); trajectory tokens preserve temporal coherence more efficiently than independent patches (TrajViT, ICCV 2025); and separating dynamic agents from static/common motion improves driving temporal modeling (DualAD, CVPR 2024).
