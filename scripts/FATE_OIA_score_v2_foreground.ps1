param(
  [string]$OutputDir = "",
  [string]$Python = "E:\Anaconda\envs\sbw39\python.exe",
  [int]$BatchSize = 4,
  [int]$GradAccum = 8,
  [int]$MaxEpochs = 20,
  [switch]$AllowTraining
)

$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\FATE_Drive\fate_oia_worktree"

$args = @(
  "-m", "fate_oia.engine.supervise_score_v2_oia",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--max_epochs", "$MaxEpochs"
)

if ($OutputDir -ne "") {
  $args += @("--output_dir", $OutputDir)
}
if ($AllowTraining) {
  $args += "--allow_training"
}

Write-Host "Starting ScoreV2 foreground supervisor"
Write-Host "Git HEAD:"
git rev-parse HEAD
& $Python @args
exit $LASTEXITCODE
