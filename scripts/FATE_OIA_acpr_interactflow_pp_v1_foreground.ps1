param(
  [int]$Epochs = 30,
  [int]$BatchSize = 4,
  [int]$GradAccum = 16,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$ReviewPass = Join-Path $Root ".background_runs\acpr_interactflow_pp_v1_preflight\REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt"
if ($RequireReviewPass -and -not (Test-Path $ReviewPass)) {
  throw "Missing REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt"
}

$Out = Join-Path $Root ".background_runs\acpr_interactflow_pp_v1_full"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_acpr_interactflow_psi `
  --config configs\acpr_interactflow_pp_v1_psi_damo_11902.yaml `
  --output_dir $Out `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --device $Device `
  --test_only `
  --no_feature_cache `
  --require_no_token_compression
