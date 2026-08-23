# TIDA Trajectory Credit V5.8 Design

## Evidence and root cause

V5.7 removed static action shortcuts, after which shuffled and reversed history had exactly zero effect and the trajectory branch reduced action mF1. The failure is in measurement: every frame independently retained only 12 action-attentive patches, and trajectory matching could only choose among those patches. A traffic entity disappeared permanently whenever it fell outside a later frame's top-12 set.

## Dense local reciprocal transport

The frozen DINO context encoder now retains its last-layer dense patch field without re-running or training the backbone. Terminal action anchors remain sparse. Each anchor is propagated backward inside a bounded local window over the complete patch field. A provisional pass estimates robust common camera displacement; a compensated second pass tracks the anchor; a reverse pass measures cycle consistency and visibility. Common motion is separated from per-track exclusive motion before the action-only trajectory head.

The target patch field is bilinearly aligned to the lower-resolution history grid. All coordinates remain normalized, and temporal interventions transform dense fields together with query and sparse-patch histories. Missing history, time reversal, and time shuffling therefore act on the actual measurement rather than only on downstream tokens.

## Safety and diagnostics

- The DINO and image baseline remain frozen and execute once per frame.
- The trajectory output projection remains zero initialized and bounded; reason logits are unchanged.
- The old sparse builder path remains as a compatibility fallback, but formal V5.8 model forward supplies dense fields.
- Reciprocal confidence is the geometric mean of provisional, backward, and forward confidence multiplied by cycle consistency. Direct multiplication was empirically over-conservative and reduced real support to 0.047.
- Artifacts report local candidate coverage, cycle confidence, exclusive/common motion ratio, ordered-control margin, decision flips, and history intervention effects.

## Validation decision

A 128-clip no-training audit must show finite dense matching, substantial candidate coverage, increased support, and non-zero history dependence. Only then may a one-epoch head-only probe train the trajectory owner on all 2,291 train-core clips. Full training is not justified unless that probe improves action metrics or produces positive target transport without damaging the strong baseline.
