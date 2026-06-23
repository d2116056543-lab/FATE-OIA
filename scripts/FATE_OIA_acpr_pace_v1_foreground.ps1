param(
  [int]$Epochs = 16,
  [int]$BatchSize = 5,
  [int]$GradAccum = 6,
  [int]$NumWorkers = 4,
  [string]$Device = "cuda",
  [string]$ReferenceCheckpoint = "",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$argsList = @(
  "-u","-m","fate_oia.engine.supervise_acpr_pace_foreground",
  "--config","configs\fate_oia_train_360x640_acpr_pace_v1.yaml",
  "--output_dir",".background_runs\acpr_pace_v1_full",
  "--epochs",$Epochs,
  "--batch_size",$BatchSize,
  "--gradient_accumulation_steps",$GradAccum,
  "--num_workers",$NumWorkers,
  "--device",$Device,
  "--reference_checkpoint",$ReferenceCheckpoint
)
if ($RequireReviewPass) { $argsList += "--require_review_pass" }
E:\Anaconda\envs\sbw39\python.exe @argsList
