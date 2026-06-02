param(
  [string]$Config = "configs\fate_oia_train_360x640_care_moe_oia_v1.yaml",
  [int]$Epochs = 24,
  [int]$BatchSize = 4,
  [int]$GradAccum = 8,
  [int]$FallbackBatchSize1 = 3,
  [int]$FallbackGradAccum1 = 11,
  [int]$FallbackBatchSize2 = 2,
  [int]$FallbackGradAccum2 = 16,
  [string]$Device = "cuda",
  [string]$BDD100KRoot = "E:\sbw\BDD100K",
  [string]$BDDOIARoot = "E:\sbw\BDD-OIA",
  [switch]$NoFeatureCache,
  [switch]$TestOnly,
  [switch]$GoalMode,
  [switch]$RequireReviewPass,
  [int]$MaxTrainSamples = 0,
  [int]$MaxTestSamples = 0
)
$ErrorActionPreference = "Stop"
Write-Host "CARE-MoE-OIA V1 foreground supervisor"
Write-Host "git commit: $(git rev-parse HEAD)"
$py = "E:\Anaconda\envs\sbw39\python.exe"
& $py -m fate_oia.engine.supervise_care_moe_oia_foreground `
  --config $Config `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --fallback_batch_size_1 $FallbackBatchSize1 `
  --fallback_grad_accum_1 $FallbackGradAccum1 `
  --fallback_batch_size_2 $FallbackBatchSize2 `
  --fallback_grad_accum_2 $FallbackGradAccum2 `
  --device $Device `
  --bdd100k_root $BDD100KRoot `
  --bdd_oia_root $BDDOIARoot `
  --require_review_pass `
  --max_train_samples $MaxTrainSamples `
  --max_test_samples $MaxTestSamples
exit $LASTEXITCODE
