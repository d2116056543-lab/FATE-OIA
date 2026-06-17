param(
  [int]$Epochs = 16,
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass,
  [string]$Resume = "",
  [switch]$SanityFineTune
)
$ErrorActionPreference = "Stop"
Set-Location E:\sbw\FATE_Drive\fate_oia_acpr_fusionlite_v1_4_worktree
$argsList = @(
  "-u", "-m", "fate_oia.engine.supervise_acpr_oia_foreground",
  "--config", "configs\fate_oia_train_360x640_acpr_fusionlite_v1_4.yaml",
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--device", "$Device",
  "--output_dir", ".background_runs\acpr_fusionlite_v1_4_360x640_testonly"
)
if ($RequireReviewPass) { $argsList += "--require_review_pass" }
if ($Resume -ne "") { $argsList += @("--resume_checkpoint", $Resume) }
if ($SanityFineTune) { $argsList += "--sanity_finetune" }
& E:\Anaconda\envs\sbw39\python.exe @argsList
