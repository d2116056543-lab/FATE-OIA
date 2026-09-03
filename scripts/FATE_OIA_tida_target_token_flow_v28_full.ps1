param(
  [string]$OutputDir = "H:\FATE_Drive_runs\tida_target_token_flow_v28_full",
  [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "E:\Anaconda\envs\sbw39\python.exe"
$config = Join-Path $repo "configs\fate_oia_train_tida_target_token_flow_v28.yaml"
$manifest = "G:\FATE_Drive_runs\tida_full_manifest_exact4572_v19\tida_full_primary_manifest.jsonl"
$checkpoint = "F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth"
$frameStores = @(
  "G:\FATE_Drive_runs\tida_raw_frames_logit_flow_v26_pilot_zip",
  "G:\FATE_Drive_runs\tida_raw_frames_primary5584_calib779_test885",
  "F:\FATE_Drive_runs\tida_raw_frames_1000_calib324_test885",
  "G:\FATE_Drive_runs\tida_raw_frames_source_complete",
  "H:\FATE_Drive_runs\tida_raw_frames_exact15_missing_v27"
)

foreach ($required in @($python, $config, $manifest, $checkpoint) + $frameStores) {
  if (-not (Test-Path -LiteralPath $required)) {
    throw "Required V28 input does not exist: $required"
  }
}
$coverage = "H:\FATE_Drive_runs\tida_target_token_flow_v27_preflight\frame15_coverage_after.json"
if (-not (Test-Path -LiteralPath $coverage)) {
  throw "Full 15-frame coverage audit is missing: $coverage"
}
$coverageResult = Get-Content -LiteralPath $coverage -Raw | ConvertFrom-Json
if (-not $coverageResult.pass -or $coverageResult.missing_available_count -ne 0) {
  throw "Full training requires history frames for every history-available case"
}
if ($coverageResult.history_unavailable_fallback_count -ne 35 -or `
    $coverageResult.history_unavailable_test_count -ne 6) {
  throw "Unexpected no-history cohort; exact image fallback contract changed"
}

if (-not $Resume -and (Test-Path -LiteralPath $OutputDir)) {
  throw "Clean full output already exists: $OutputDir"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$env:PYTHONPATH = $repo
Set-Location -LiteralPath $repo

$arguments = @(
  "-u", "-m", "fate_oia.engine.train_tida_oia",
  "--config", $config,
  "--clip-manifest", $manifest,
  "--image-checkpoint", $checkpoint,
  "--frame-store-root", ($frameStores -join ";"),
  "--output-dir", $OutputDir,
  "--epochs", "10",
  "--batch-size", "6",
  "--gradient-accumulation-steps", "5",
  "--context-chunk-size", "3",
  "--num-workers", "4",
  "--train-owners", "target_token_action,target_token_reason",
  "--eval-every-epochs", "1",
  "--run-kind", "full",
  "--skip-ema-eval",
  "--skip-expanded-eval",
  "--device", "cuda"
)
if ($Resume) {
  if (-not (Test-Path -LiteralPath $Resume)) {
    throw "Resume checkpoint does not exist: $Resume"
  }
  $arguments += @("--resume", $Resume)
}

$ErrorActionPreference = "Continue"
& $python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "full_train.log")
$exitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Set-Content -LiteralPath (Join-Path $OutputDir "process_exit_code.txt") -Value $exitCode -Encoding ascii
exit $exitCode
