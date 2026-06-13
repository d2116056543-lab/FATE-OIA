param(
  [int]$Epochs = 24,
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
cd E:\sbw\FATE_Drive\fate_oia_eagle_pu_v1_worktree
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = ".background_runs\eagle_pu_v1_full_$stamp"
$args = @("-u", "-m", "fate_oia.engine.supervise_eagle_pu_foreground", "--config", "configs\fate_oia_train_360x640_eagle_pu_v1.yaml", "--output_dir", $out, "--epochs", "$Epochs", "--batch_size", "$BatchSize", "--gradient_accumulation_steps", "$GradAccum", "--device", $Device)
if ($RequireReviewPass) { $args += "--require_review_pass" }
& E:\Anaconda\envs\sbw39\python.exe @args
exit $LASTEXITCODE
