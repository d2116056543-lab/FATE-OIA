param(
  [int]$Epochs = 18,
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = "E:\Anaconda\envs\sbw39\python.exe"
$Config = "configs\fate_oia_train_360x640_acpr_actalign_v1_3.yaml"
$Preflight = ".background_runs\acpr_actalign_v1_3_preflight"
$ReviewPass = Join-Path $Preflight "REVIEW_PASS_ACPR_ACTALIGN_V1_3.txt"
if ($RequireReviewPass -and -not (Test-Path $ReviewPass)) {
  & $Python -m fate_oia.engine.audit_acpr_oia_implementation --config $Config --output_dir $Preflight --device $Device --write_review_pass
  if (-not (Test-Path $ReviewPass)) { throw "Missing REVIEW_PASS_ACPR_ACTALIGN_V1_3.txt" }
}
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Output = ".background_runs\acpr_actalign_v1_3_full_$stamp"
Write-Host "ACPR-ActAlign V1.3 foreground training"
Write-Host "config=$Config output=$Output epochs=$Epochs batch=$BatchSize accum=$GradAccum device=$Device"
& $Python -u -m fate_oia.engine.train_acpr_oia `
  --config $Config `
  --output_dir $Output `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --device $Device `
  --test_only `
  --no_feature_cache `
  --require_no_token_compression
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
