param(
  [int]$Epochs = 28,
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$argsList = @(
  "-u", "-m", "fate_oia.engine.supervise_acpr_oia_foreground",
  "--config", "configs\fate_oia_train_360x640_acpr_oia_v1.yaml",
  "--output_dir", ".background_runs\acpr_oia_v1_full",
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--device", "$Device"
)
if ($RequireReviewPass) { $argsList += "--require_review_pass" }
& E:\Anaconda\envs\sbw39\python.exe @argsList
exit $LASTEXITCODE
