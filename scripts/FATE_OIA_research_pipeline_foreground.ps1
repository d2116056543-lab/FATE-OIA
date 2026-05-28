param(
  [string]$Root = "E:\sbw\FATE_Drive",
  [string]$Repo = "E:\sbw\FATE_Drive\fate_oia_worktree",
  [string]$Python = "E:\Anaconda\envs\sbw39\python.exe",
  [int]$BatchSize = 4,
  [int]$GradientAccumulationSteps = 8,
  [int]$MinGateEpoch = 14,
  [string]$PretrainedWeights = "ckp\reference\dino_deitsmall8_pretrain.pth"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repo
$staged = git status --short
if ($staged -match "\.background_runs") {
  throw ".background_runs appears in git status; refusing to start foreground research pipeline."
}
$runRoot = ".background_runs\oia_research_pipeline_$(Get-Date -Format yyyyMMdd_HHmmss)"
$cmd = @(
  $Python, "-m", "fate_oia.engine.supervise_oia_research_pipeline",
  "--root", $Root,
  "--repo", $Repo,
  "--python", $Python,
  "--run_root", $runRoot,
  "--pretrained_weights", $PretrainedWeights,
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradientAccumulationSteps",
  "--min_gate_epoch", "$MinGateEpoch",
  "--allow_training"
)
Write-Host "Foreground command:"
Write-Host ($cmd -join " ")
& $cmd[0] $cmd[1..($cmd.Count-1)]
