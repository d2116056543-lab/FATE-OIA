param(
  [int]$Epochs = 16,
  [int]$NumWorkers = 6,
  [string]$Device = "cuda",
  [string]$ReferenceCheckpoint = "",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
cd E:\sbw\FATE_Drive\fate_oia_acpr_gem_v1_worktree
$argsList = @(
  "-u", "-m", "fate_oia.engine.supervise_acpr_gem_foreground",
  "--epochs", "$Epochs",
  "--num_workers", "$NumWorkers",
  "--device", "$Device",
  "--reference_checkpoint", "$ReferenceCheckpoint"
)
if ($RequireReviewPass) { $argsList += "--require_review_pass" }
& E:\Anaconda\envs\sbw39\python.exe @argsList
