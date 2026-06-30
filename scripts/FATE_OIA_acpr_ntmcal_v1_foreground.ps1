param(
  [int]$Epochs = 18,
  [int]$BatchSize = 8,
  [int]$GradAccum = 4,
  [int]$NumWorkers = 8,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
$OutputDir = ".background_runs\acpr_ntmcal_v1_360x640_testprimary"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1 -Device $Device
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_ntmcal_implementation --config configs\fate_oia_train_360x640_acpr_ntmcal_v1.yaml --output_dir .background_runs\acpr_ntmcal_v1_preflight --device $Device --write_review_pass
if ($RequireReviewPass -and -not (Test-Path ".background_runs\acpr_ntmcal_v1_preflight\REVIEW_PASS_ACPR_NTMCAL_V1.txt")) {
  throw "missing REVIEW_PASS_ACPR_NTMCAL_V1.txt"
}
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_acpr_ntmcal_oia `
  --config configs\fate_oia_train_360x640_acpr_ntmcal_v1.yaml `
  --output_dir $OutputDir `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --num_workers $NumWorkers `
  --device $Device `
  --amp_dtype bf16 `
  --test_only `
  --best_selection_split test `
  --best_selection_metric joint_test_score `
  --no_feature_cache `
  --token_compression none `
  --require_no_token_compression `
  --require_review_pass .background_runs\acpr_ntmcal_v1_preflight\REVIEW_PASS_ACPR_NTMCAL_V1.txt
