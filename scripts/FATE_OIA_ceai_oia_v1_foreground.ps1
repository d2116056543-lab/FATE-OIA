param(
  [string]$Config = "configs\fate_oia_train_360x640_ceai_oia_v1.yaml",
  [int]$Epochs = 32,
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
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
$py = "E:\Anaconda\envs\sbw39\python.exe"
$cmd = @(
  "-m", "fate_oia.engine.supervise_ceai_oia_foreground",
  "--config", $Config,
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--fallback_batch_size1", "$FallbackBatchSize1",
  "--fallback_gradient_accumulation_steps1", "$FallbackGradAccum1",
  "--fallback_batch_size2", "$FallbackBatchSize2",
  "--fallback_gradient_accumulation_steps2", "$FallbackGradAccum2",
  "--device", $Device,
  "--bdd100k_root", $BDD100KRoot,
  "--bdd_oia_root", $BDDOIARoot
)
if ($RequireReviewPass) { $cmd += "--require_review_pass" }
if ($NoFeatureCache) { $cmd += "--no_feature_cache" }
if ($TestOnly) { $cmd += "--test_only" }
if ($GoalMode) { $cmd += "--goal_mode" }
& $py @cmd
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
