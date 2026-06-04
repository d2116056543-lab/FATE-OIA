param(
  [string]$Config = "configs\fate_oia_train_360x640_p3le_pair_oia_v1.yaml",
  [string]$Device = "cuda",
  [int]$Epochs = 28,
  [int]$BatchSize = 4,
  [Alias("GradAccum")]
  [int]$GradientAccumulationSteps = 8,
  [int]$FallbackBatchSize1 = 3,
  [Alias("FallbackGradAccum1")]
  [int]$FallbackGradientAccumulationSteps1 = 11,
  [int]$FallbackBatchSize2 = 2,
  [Alias("FallbackGradAccum2")]
  [int]$FallbackGradientAccumulationSteps2 = 16,
  [string]$BDD100KRoot = "E:\sbw\BDD100K",
  [string]$BDDOIARoot = "E:\sbw\BDD-OIA",
  [int]$NumWorkers = 4,
  [int]$MaxTrainSamples = 0,
  [int]$MaxTestSamples = 0,
  [switch]$NoFeatureCache,
  [switch]$TestOnly,
  [switch]$GoalMode,
  [switch]$RequireReviewPass
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
if (-not $NoFeatureCache) { Write-Host "NoFeatureCache switch not provided; config/audit still enforces no cache." }
if (-not $TestOnly) { Write-Host "TestOnly switch not provided; config/audit still enforces test-only." }
if (-not $GoalMode) { Write-Host "GoalMode switch not provided; supervisor still writes goal completion artifact." }

& $Python -m fate_oia.engine.supervise_p3le_pair_oia_foreground `
  --config $Config `
  --device $Device `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradientAccumulationSteps `
  --fallback_batch_size1 $FallbackBatchSize1 `
  --fallback_gradient_accumulation_steps1 $FallbackGradientAccumulationSteps1 `
  --fallback_batch_size2 $FallbackBatchSize2 `
  --fallback_gradient_accumulation_steps2 $FallbackGradientAccumulationSteps2 `
  --num_workers $NumWorkers `
  --bdd100k_root $BDD100KRoot `
  --bdd_oia_root $BDDOIARoot `
  --max_train_samples $MaxTrainSamples `
  --max_test_samples $MaxTestSamples `
  --require_review_pass
