param(
  [int]$Epochs = 18,
  [int]$BatchSize = 9,
  [int]$GradAccum = 4,
  [string]$Device = "cuda",
  [int]$TargetGpuGB = 45,
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = ".background_runs\acpr_pmcal_v2_direct_image_$RunId"
$ReviewPass = ".background_runs\acpr_pmcal_v2_preflight\REVIEW_PASS_PMCalV2.txt"
if ($RequireReviewPass -and -not (Test-Path $ReviewPass)) {
  throw "Missing review pass: $ReviewPass"
}
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_pmcal_v2_foreground `
  --config configs\fate_oia_train_360x640_acpr_pmcal_v2.yaml `
  --output_dir $OutputDir `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --device $Device `
  --require_review_pass `
  --review_pass_path $ReviewPass
exit $LASTEXITCODE
