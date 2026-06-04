param(
  [string]$Config = "configs\fate_oia_train_360x640_p3le_pair_oia_v1.yaml",
  [string]$Device = "cuda",
  [int]$Epochs = 28,
  [int]$BatchSize = 4,
  [int]$GradientAccumulationSteps = 8,
  [int]$NumWorkers = 4,
  [int]$MaxTrainSamples = 0,
  [int]$MaxTestSamples = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = "E:\Anaconda\envs\sbw39\python.exe"
Write-Host "P3LE-PAIR-OIA V1 foreground supervisor"
Write-Host "Repo: $RepoRoot"
Write-Host "Git HEAD:"
git rev-parse HEAD
Write-Host "Config: $Config"
Write-Host "Evaluation: test-only; best checkpoint: test joint"
Write-Host "Forbidden: feature cache, token compression, val-selected best, background process"

& $Python -m fate_oia.engine.supervise_p3le_pair_oia_foreground `
  --config $Config `
  --device $Device `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradientAccumulationSteps `
  --fallback_batch_size1 3 `
  --fallback_gradient_accumulation_steps1 11 `
  --fallback_batch_size2 2 `
  --fallback_gradient_accumulation_steps2 16 `
  --num_workers $NumWorkers `
  --max_train_samples $MaxTrainSamples `
  --max_test_samples $MaxTestSamples `
  --require_review_pass
