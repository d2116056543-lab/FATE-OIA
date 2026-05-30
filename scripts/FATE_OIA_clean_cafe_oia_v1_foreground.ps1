param(
  [string]$Worktree = "E:\sbw\FATE_Drive\fate_oia_clean_cafe_oia_v1_worktree",
  [string]$OutputName = "clean_cafe_oia_v1_360x640",
  [int]$Epochs = 40,
  [int]$BatchSize = 2,
  [int]$GradAccum = 16,
  [int]$MaxTrainSamples = 0,
  [int]$MaxValSamples = 0,
  [int]$MaxTestSamples = 0,
  [string]$ResumeCheckpoint = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $Worktree
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $Worktree ".background_runs\$OutputName`_$ts"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$argsList = @(
  "-u", "-m", "fate_oia.engine.supervise_cafe_oia_foreground",
  "--config", "configs\fate_oia_train_360x640_cafe_oia_v1.yaml",
  "--output_dir", $out,
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--foreground",
  "--require_review_pass",
  "--device", "cuda"
)
if ($MaxTrainSamples -gt 0) { $argsList += @("--max_train_samples", "$MaxTrainSamples") }
if ($MaxValSamples -gt 0) { $argsList += @("--max_val_samples", "$MaxValSamples") }
if ($MaxTestSamples -gt 0) { $argsList += @("--max_test_samples", "$MaxTestSamples") }
if ($ResumeCheckpoint) { $argsList += @("--resume_checkpoint", $ResumeCheckpoint) }

Write-Host "CAFE foreground output: $out"
& E:\Anaconda\envs\sbw39\python.exe @argsList
