param(
  [int]$Epochs = 16,
  [int]$NumWorkers = 6,
  [string]$Device = "cuda",
  [string]$ReferenceCheckpoint = "",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$argsList = @(
  "-m", "fate_oia.engine.supervise_acpr_vista_foreground",
  "--epochs", "$Epochs",
  "--num_workers", "$NumWorkers",
  "--device", "$Device",
  "--reference_checkpoint", "$ReferenceCheckpoint"
)
if ($RequireReviewPass) { $argsList += "--require_review_pass" }
& E:\Anaconda\envs\sbw39\python.exe @argsList

