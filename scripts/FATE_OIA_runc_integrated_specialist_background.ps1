param(
  [string]$OutputDir = "",
  [int]$Epochs = 24,
  [int]$BatchSize = 4,
  [int]$GradientAccumulationSteps = 8,
  [int]$MaxTrainSamples = 0,
  [int]$MaxValSamples = 0,
  [int]$MaxTestSamples = 0
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
$argsList = @(
  "-m", "fate_oia.engine.supervise_runc_integrated_specialist",
  "--launch_training",
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradientAccumulationSteps",
  "--max_train_samples", "$MaxTrainSamples",
  "--max_val_samples", "$MaxValSamples",
  "--max_test_samples", "$MaxTestSamples"
)
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
  $argsList += @("--output_dir", $OutputDir)
}
& "E:\Anaconda\envs\sbw39\python.exe" @argsList
exit $LASTEXITCODE
