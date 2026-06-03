param(
  [string]$Config = "configs\fate_oia_train_360x640_psr_train_oia_v1.yaml",
  [string]$Device = "cuda",
  [int]$Epochs = 24,
  [int]$BatchSize = 4,
  [int]$GradientAccumulationSteps = 8,
  [int]$NumWorkers = 4,
  [int]$MaxTrainSamples = 0,
  [int]$MaxTestSamples = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot
Write-Host "PSR-Train OIA V1 foreground supervisor"
Write-Host "Repo: $RepoRoot"
Write-Host "Config: $Config"
Write-Host "Device: $Device"
Write-Host "Batch/Accum: $BatchSize/$GradientAccumulationSteps"
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.supervise_psr_train_oia_foreground `
  --config $Config `
  --device $Device `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradientAccumulationSteps `
  --num_workers $NumWorkers `
  --max_train_samples $MaxTrainSamples `
  --max_test_samples $MaxTestSamples `
  --require_audit
