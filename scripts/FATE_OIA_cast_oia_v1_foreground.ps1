param(
  [int]$Epochs = 40,
  [int]$BatchSize = 5,
  [int]$GradAccum = 6,
  [string]$Device = "cuda",
  [string]$ResumeCheckpoint = "",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
cd (Split-Path -Parent $PSScriptRoot)
$argsList = @(
  "-u", "-m", "fate_oia.engine.supervise_cast_oia_foreground",
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--grad_accum", "$GradAccum",
  "--device", "$Device",
  "--output_dir", ".background_runs/cast_oia_v1_full"
)
if ($RequireReviewPass) { $argsList += "--require_review_pass" }
if ($ResumeCheckpoint -ne "") { $argsList += @("--resume_checkpoint", "$ResumeCheckpoint") }
& E:\Anaconda\envs\sbw39\python.exe @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
