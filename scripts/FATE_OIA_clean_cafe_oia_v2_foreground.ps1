param(
  [int]$Epochs = 24,
  [int]$BatchSize = 4,
  [int]$GradAccum = 8,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass,
  [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$Repo = "E:\sbw\FATE_Drive\fate_oia_clean_cafe_oia_v1_worktree"
Set-Location $Repo

if (-not $OutputDir) {
  $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $OutputDir = ".background_runs\clean_cafe_oia_v2_evidence_fixed_360x640_$Stamp"
}

$ArgsList = @(
  "-u", "-m", "fate_oia.engine.supervise_cafe_oia_v2_foreground",
  "--config", "configs\fate_oia_train_360x640_cafe_oia_v2.yaml",
  "--output_dir", $OutputDir,
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--device", $Device,
  "--foreground"
)
if ($RequireReviewPass) {
  $ArgsList += "--require_review_pass"
}

Write-Host "CAFE-OIA V2 foreground supervisor"
Write-Host "Repo: $Repo"
Write-Host "Output: $OutputDir"
Write-Host "Command: E:\Anaconda\envs\sbw39\python.exe $($ArgsList -join ' ')"
& E:\Anaconda\envs\sbw39\python.exe @ArgsList
exit $LASTEXITCODE
