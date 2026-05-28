param(
  [string]$RunCDir = "E:\sbw\FATE_Drive\fate_oia_worktree\.background_runs\fate_oia_runC_e13_cosine_labelcorr_20260526_191938",
  [string]$OutputRoot = "E:\sbw\FATE_Drive\fate_oia_worktree\.background_runs",
  [string]$Python = "E:\Anaconda\envs\sbw39\python.exe",
  [string[]]$Stages = @("P0", "P1", "P2", "P5"),
  [int]$P1Epochs = 4,
  [int]$P2Epochs = 8,
  [int]$BatchSize = 512,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Set-Location "E:\sbw\FATE_Drive\fate_oia_worktree"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = Join-Path $OutputRoot "fate_oia_tail_adapter_$stamp"

Write-Host "FATE-OIA tail-adapter foreground supervisor"
Write-Host "Run C dir: $RunCDir"
Write-Host "Output dir: $outputDir"
Write-Host "Stages: $($Stages -join ',')"
Write-Host "Git HEAD:"
git rev-parse HEAD

$args = @(
  "-m", "fate_oia.engine.supervise_tail_adapter_oia",
  "--run_c_dir", $RunCDir,
  "--output_dir", $outputDir,
  "--stages"
) + $Stages + @(
  "--p1_epochs", "$P1Epochs",
  "--p2_epochs", "$P2Epochs",
  "--batch_size", "$BatchSize"
)

if ($DryRun) {
  $args += "--dry_run"
}

& $Python @args
exit $LASTEXITCODE
