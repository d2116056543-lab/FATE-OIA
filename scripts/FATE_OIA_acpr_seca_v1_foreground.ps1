param(
  [int]$Epochs = 14,
  [int]$BatchSize = 5,
  [int]$GradAccum = 6,
  [int]$NumWorkers = 4,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$argsList = @(
  "-u", "-m", "fate_oia.engine.supervise_acpr_seca_foreground",
  "--config", "configs\fate_oia_train_360x640_acpr_seca_v1.yaml",
  "--output_dir", ".background_runs\acpr_seca_v1_full",
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--num_workers", "$NumWorkers",
  "--device", "$Device"
)
if ($RequireReviewPass) { $argsList += "--require_review_pass" }
& E:\Anaconda\envs\sbw39\python.exe @argsList
exit $LASTEXITCODE
