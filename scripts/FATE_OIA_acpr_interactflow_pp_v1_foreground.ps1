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

$RuntimeRoot = "E:\sbw\runtime_cache"
$TempRoot = Join-Path $RuntimeRoot "tmp"
$HfRoot = "E:\sbw\hf_cache"
New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $HfRoot | Out-Null
$env:TMP = $TempRoot
$env:TEMP = $TempRoot
$env:HF_HOME = $HfRoot
$env:TRANSFORMERS_CACHE = Join-Path $HfRoot "transformers"
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"
$env:USE_TF = "0"
$env:USE_FLAX = "0"

$ReviewPass = Join-Path $Root ".background_runs\acpr_interactflow_pp_v1_preflight\REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt"
if ($RequireReviewPass -and -not (Test-Path $ReviewPass)) {
  throw "Missing REVIEW_PASS_ACPR_INTERACTFLOW_PP_V1.txt"
}
if ($RequireReviewPass) {
  $LocalHead = (git rev-parse HEAD).Trim()
  $Review = Get-Content -LiteralPath $ReviewPass -Raw | ConvertFrom-Json
  if ($Review.git_head -ne $LocalHead) {
    throw "Stale REVIEW_PASS: review git_head=$($Review.git_head), local HEAD=$LocalHead"
  }
  $Dirty = git status --porcelain
  if ($Dirty) {
    throw "Worktree is dirty; commit code-only changes and rerun preflight before full train."
  }
  $RemoteLine = git ls-remote github refs/heads/acpr_interactflow_pp_v1
  if (-not $RemoteLine) {
    throw "Cannot verify GitHub branch refs/heads/acpr_interactflow_pp_v1"
  }
  $RemoteHead = ($RemoteLine -split "\s+")[0]
  if ($RemoteHead -ne $LocalHead) {
    throw "GitHub branch HEAD mismatch: remote=$RemoteHead local=$LocalHead"
  }
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
