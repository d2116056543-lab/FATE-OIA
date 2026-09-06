# COEV-OIA v1 Implementation Coverage Plan

Canonical specification: the six UTF-8 documents supplied on 2026-09-06. This
implementation stays on `tida_site_v20`, starting at
`f33d7d5a52848bf52f41b844f865588ab49f1672`. It must not invoke the legacy TIDA
builder or load an old task checkpoint.

## Evidence rule

Every requirement needs four kinds of evidence before it can be marked complete:

1. a production symbol imported by the formal factory;
2. a focused test that fails when the behavior is removed;
3. an observed runtime artifact from the production factory;
4. a supervisor review of the actual call path and gradient owner.

`PASS` text, file existence, a mock forward, or a shape-only assertion is not
sufficient.

## Requirement coverage matrix

| ID | Production responsibility | Test / mutation evidence | Runtime artifact |
|---|---|---|---|
| R01 | `coev_preflight.git_identity` | exact branch/worktree test | `data_audit.json` identity |
| R02 | `build_coev_model` fresh task state | reject task checkpoint mutation | implementation audit |
| R03 | `CoEVVideoDataset` raw RGB decode | cache/store I/O trap | I/O trace |
| R04 | PTS-aware frame sampler | VFR/repeated-index tests | dt quantiles |
| R05 | ID/label/flip contracts | two-flip restoration | ordered ID hashes |
| R06 | `CoEVInputs`/`CoEVTargets` split | poisoned annotation read | test forward trace |
| R07 | `CoEVVisualField` prefix/upper split | per-block grad/update probe | owner gradients |
| R08 | true blocks 4/8/12 | forward hooks and dependency mutation | hook trace |
| R09 | history task gradients | early/middle/late frame probes | temporal gradient JSONL |
| R10 | 25-query decoder and full patch reread | 86,064 K/V and sensitivity | decoder stats |
| R11 | eight independent predicate maps | source/unknown/sigmoid tests | observer coverage |
| R12 | detached task maps, live grounding maps | forbidden owner gradients | owner matrix |
| R13 | bidirectional matcher+dustbin | synthetic warp/occlusion | matcher stats |
| R14 | five-step IRLS background affine | camera/object/pathology fixtures | affine stats |
| R15 | physical `actual_dt` | same displacement, different dt | dt/motion stats |
| R16 | 14 primitive coordinates | source/unit/valid-zero tests | primitive stats |
| R17 | persistent-state features | constant-path oracle | lift stats |
| R18 | directed area | forward/reverse square oracle | area stats |
| R19 | exact subdivision invariance | segmented line oracle | math audit |
| R20 | missing/window semantics | no-gap and interpolation tests | coverage stats |
| R21 | explicit time coordinate | duration sensitivity test | time feature stats |
| R22 | named basis locality | Jacobian/mutation test | locality audit |
| R23 | additive readout accounting | reconstruction/deletion tests | contribution audit |
| R24 | sole final `logits` view | output-bypass mutation | call-path audit |
| R25 | all losses and exact optimizer ownership | per-loss autograd/update | owner matrix |
| R26 | uncapped evidence gradients | cap/global-scale forbidden scan | RMS/cancellation stats |
| R27 | full recomputation interventions | forward-call counter | intervention manifest |
| R28 | one full test eval per epoch | eval call trace | epoch metrics |
| R29 | raw fixed-0.5 joint selection | metric/tie fixtures | best manifest |
| R30 | no test feedback | label-poison state comparison | feedback audit |
| R31 | formal update scheduler/tail weighting | tail/smoke schedule tests | update trace |
| R32 | boundary-exact resume | uninterrupted/resumed fixture | resume audit |
| R33 | real RGB/BF16 GPU profile | 200-update stress | GPU profile |
| R34 | attached foreground supervisor | parent/child/exit tests | supervisor log |
| R35 | strict completion semantics | interrupted-run mutation | completion audit |
| R36 | tracked files and remote identity | `git ls-files/ls-remote` | git audit |
| R37 | mutation suite | at least 10 shape-valid kills | kill report |
| R38 | interpretation boundary | schema/text assertions | final diagnostics |
| R39 | atomic bounded artifacts | disk/failure/load tests | artifact inventory |
| R40 | source/config/spec/skill/data binding | hash mismatch mutations | ready manifest |

## Checkpoints

- Mathematical review: data identity, time, missingness, path lift, readout.
- Architectural review: DINO graph, observer isolation, full-field decoder.
- Runtime review: formal factory, optimizer ownership, artifacts, resume, GPU.

No full training is allowed until all three reviews have concrete evidence and the
clean pushed HEAD is rebound into a new ready manifest.
