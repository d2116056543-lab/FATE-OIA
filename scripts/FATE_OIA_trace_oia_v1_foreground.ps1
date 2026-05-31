param(
  [int]$Epochs=20,
  [int]$BatchSize=4,
  [int]$GradAccum=8,
  [int]$FallbackBatchSize1=3,
  [int]$FallbackGradAccum1=11,
  [int]$FallbackBatchSize2=2,
  [int]$FallbackGradAccum2=16,
  [string]$Device="cuda",
  [switch]$RequireReviewPass,
  [string]$OutputDir="",
  [string]$ReviewPassPath=".background_runs\trace_action_primary_v2_preflight\REVIEW_PASS_TRACE_ACTION_PRIMARY.txt",
  [int]$MaxTrainSamples=0,
  [int]$MaxTestSamples=0
)
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
if ($OutputDir -eq "") {
  $stamp=Get-Date -Format "yyyyMMdd_HHmmss"
  $OutputDir=".background_runs\trace_action_primary_v2_direct_image_$stamp"
}
$args=@(
  "-m","fate_oia.engine.supervise_trace_oia_foreground",
  "--config","configs\fate_oia_train_360x640_trace_action_primary_v2.yaml",
  "--output_dir",$OutputDir,
  "--epochs","$Epochs",
  "--batch_size","$BatchSize",
  "--grad_accum","$GradAccum",
  "--fallback_batch_size_1","$FallbackBatchSize1",
  "--fallback_grad_accum_1","$FallbackGradAccum1",
  "--fallback_batch_size_2","$FallbackBatchSize2",
  "--fallback_grad_accum_2","$FallbackGradAccum2",
  "--device",$Device,
  "--review_pass_path",$ReviewPassPath,
  "--max_train_samples","$MaxTrainSamples",
  "--max_test_samples","$MaxTestSamples",
  "--disable_feature_cache",
  "--skip_cache_build"
)
if ($RequireReviewPass) { $args += "--require_review_pass" }
& E:\Anaconda\envs\sbw39\python.exe @args
exit $LASTEXITCODE
