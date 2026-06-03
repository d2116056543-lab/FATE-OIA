param(
  [string]$RegistryConfig = "configs\psr_oia_v2_registry.yaml",
  [string]$RouterConfig = "configs\psr_oia_v2_router.yaml",
  [string]$Device = "cuda",
  [string]$BDD100KRoot = "E:\sbw\BDD100K",
  [string]$BDDOIARoot = "E:\sbw\BDD-OIA",
  [switch]$NoFeatureCache,
  [switch]$TestOnly,
  [switch]$GoalMode,
  [switch]$RequireReviewPass
)

$ErrorActionPreference = "Stop"
$Python = "E:\Anaconda\envs\sbw39\python.exe"

Write-Host "[PSR-OIA V2] foreground supervisor attached"
Write-Host "[PSR-OIA V2] registry=$RegistryConfig router=$RouterConfig device=$Device"
Write-Host "[PSR-OIA V2] BDD100K=$BDD100KRoot BDD-OIA=$BDDOIARoot no_feature_cache=$NoFeatureCache test_only=$TestOnly goal_mode=$GoalMode"

$argsList = @(
  "-m", "fate_oia.engine.supervise_psr_oia_goal",
  "--registry_config", $RegistryConfig,
  "--router_config", $RouterConfig,
  "--device", $Device
)

if ($RequireReviewPass) {
  $argsList += "--require_review_pass"
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
  throw "PSR-OIA V2 foreground supervisor failed with exit code $LASTEXITCODE"
}
