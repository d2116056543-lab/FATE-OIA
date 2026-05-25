# FATE-OIA Implementation Status 2026-05-25

## Required Context

Before starting any remote FATE-OIA task, read:

- `E:\sbw\FATE_Drive\task_plan.md`
- `E:\sbw\FATE_Drive\findings.md`
- `E:\sbw\FATE_Drive\progress.md`

## What Is Implemented

FATE-OIA now contains both the SNNA25 baseline path and the FATE-OIA full token-model path.

Implemented modules:

- BDD-OIA action/reason loader: `fate_oia/datasets/bdd_oia_multitask.py`
- BDD100K grounding index: `fate_oia/datasets/bdd100k_grounding.py`
- SNNA25 baseline head: `fate_oia/models/snna25_head.py`
- Full token model: `fate_oia/models/fate_oia_model.py`
- Label-query head: `fate_oia/models/label_query_head.py`
- Reason-to-action bottleneck: `fate_oia/models/reason_to_action_bottleneck.py`
- Token compression/provenance: `fate_oia/models/token_compressor.py`, `fate_oia/models/token_provenance.py`
- Grounding masks/losses: `fate_oia/grounding/mask_builder.py`, `fate_oia/grounding/losses.py`
- SNNA++ helper: `fate_oia/explain/snna_plus.py`
- SNNA25 train/eval: `fate_oia/engine/train_snna25.py`, `fate_oia/engine/eval_snna25.py`
- Full FATE-OIA train: `fate_oia/engine/train_fate_oia.py`
- Grounding cache build/audit: `fate_oia/engine/build_grounding_cache.py`, `fate_oia/engine/audit_grounding_cache.py`
- Counterfactual eval scaffold: `fate_oia/engine/eval_counterfactual.py`

## Full Model Training Path

`train_fate_oia.py` uses ViT token features rather than only CLS features:

```text
image -> frozen SNNA/DINO ViT token features
      -> optional keep+merge token compression with provenance
      -> FATEOIAFeatureModel label-query head
      -> action_logits + reason_logits
      -> reason_to_action_logits
      -> ASL/BCE action+reason loss
      -> reason-to-action consistency loss
      -> optional grounding loss from BDD100K object boxes
      -> optional counterfactual deletion loss
```

Smoke command:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.train_fate_oia ^
  --output_dir .background_runs\fate_oia_full_grounding_smoke_20260525 ^
  --pretrained_weights ckp\reference\dino_deitsmall8_pretrain.pth ^
  --epochs 1 --max_train_samples 2 --max_val_samples 2 ^
  --batch_size 1 --num_workers 0 --log_every 1 ^
  --token_keep_ratio 0.5 --num_summary_tokens 2 --min_tokens 4 ^
  --loss_grounding 0.01 ^
  --grounding_cache_jsonl .background_runs\fate_oia_grounding_cache_20260525.jsonl ^
  --device cuda
```

Verified smoke output:

- emitted `fate_oia_batch` for train and val
- token compression active: `785 -> 395` tokens
- grounding loss active and nonzero in batch logs
- emitted `fate_oia_epoch`

## Grounding Coverage

Full grounding cache was built and audited on the current remote data.

Coverage:

- train: label/drivable `15165 / 16082 = 94.30%`, semantic seg `854 / 16082 = 5.31%`
- val: label/drivable `2152 / 2270 = 94.80%`, semantic seg `111 / 2270 = 4.89%`
- test: label/drivable `4351 / 4572 = 95.17%`, semantic seg `263 / 4572 = 5.75%`

Important boundary:

- Object box and drivable map grounding can be used broadly.
- Semantic segmentation is subset-only and must not be claimed as full-split GT.

## Verification

Fresh GitHub zip compile:

- `COMPILED_FILES 46`
- `FRESH_ZIP_PY_COMPILE_OK`

Remote tests:

```powershell
E:\Anaconda\envs\sbw39\python.exe -m pytest ^
  tests/test_bdd_oia_dataset.py ^
  tests/test_fate_oia_metrics.py ^
  tests/test_token_provenance.py ^
  tests/test_fate_oia_snna25.py ^
  tests/test_fate_oia_grounding_counterfactual.py ^
  tests/test_fate_oia_full_training_utils.py -q
```

Result:

```text
15 passed
```

## What Is Still Not A Final Result

The full method code path is implemented and smoke-tested, but final paper-style results still require:

- full training on train split
- final `checkpoint_best.pth` / `checkpoint_latest.pth`
- full val/test action/reason metrics
- trained-checkpoint grounding and counterfactual faithfulness reports

## 2026-05-25 Gap Closure Update

The latest GPTPro audit identified two remaining FATE-OIA code-level gaps:

- counterfactual deletion loss needed an explicit gradient verification;
- grounding needed to be label-conditioned by reason-specific BDD100K category rules, not only a global average-attention mask.

Both are now implemented and verified.

### New/Updated Code

- `fate_oia/engine/train_fate_oia.py`
  - Added `--grounding_mode global|label|both`, default `both`.
  - Added `--reason_grounding_rules`, default `configs/reason_grounding_rules.yaml`.
  - Added robust loading of reason-to-BDD100K category mappings.
  - Label-conditioned grounding uses attention label index `action_dim + reason_idx` and only applies categories mapped to the active reason label.
  - Batch logs now include per-reason grounding stats, for example `reason_3_count` and `reason_4_count`.

- `tests/test_fate_oia_full_training_utils.py`
  - Added `counterfactual_deletion_loss` gradient test.
  - Added label-conditioned grounding test with a positive reason-to-person mask.

### Verification

Targeted tests:

```text
tests/test_fate_oia_full_training_utils.py: 6 passed
```

Broader selected tests:

```text
17 passed
```

Real full-model smoke:

```text
event=fate_oia_batch train=true step=0
event=fate_oia_batch train=true step=1
event=fate_oia_batch train=false step=0
event=fate_oia_batch train=false step=1
event=fate_oia_epoch epoch=0
token compression: 785 original tokens -> 395 reduced tokens
label-conditioned grounding observed:
  reason_3_count = 1
  reason_4_count = 1
```

Fresh GitHub clone compile:

```text
FATE-OIA commit 63a9c33ad4801b327d81637e623260abe51a7d84
py_compile PASS
```

Boundary remains unchanged: this is code/smoke completeness, not final trained-result completeness.
