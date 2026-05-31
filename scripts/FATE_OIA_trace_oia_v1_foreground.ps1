param([int]$Epochs=20,[int]$BatchSize=8,[int]$GradAccum=4,[int]$FallbackBatchSize=4,[int]$FallbackGradAccum=8,[int]$CacheBatchSize=8,[string]$Device="cuda",[switch]$RequireReviewPass,[string]$OutputDir="",[string]$ReviewPassPath=".background_runs\trace_oia_v1_preflight_final_head\REVIEW_PASS_TRACE_OIA.txt",[switch]$SkipCacheBuild,[int]$MaxTrainSamples=0,[int]$MaxTestSamples=0)
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
$active = Get-CimInstance Win32_Process | Where-Object {
  $_.ProcessId -ne $PID -and
  $_.CommandLine -match 'fate_oia\.engine\.(supervise_trace_oia_foreground|train_trace_oia|build_trace_oia_token_cache)'
}
if ($active) {
  $active | Select-Object ProcessId,Name,CommandLine | Format-List
  throw "Existing TRACE-OIA python process detected; refusing to start a second run."
}
if ($OutputDir -eq "") { $stamp=Get-Date -Format "yyyyMMdd_HHmmss"; $OutputDir=".background_runs\trace_oia_v1_proto_transport_360x640_cache_fulltoken_$stamp" }
$args=@("-m","fate_oia.engine.supervise_trace_oia_foreground","--config","configs\fate_oia_train_360x640_trace_oia_v1.yaml","--output_dir",$OutputDir,"--epochs","$Epochs","--batch_size","$BatchSize","--grad_accum","$GradAccum","--fallback_batch_size","$FallbackBatchSize","--fallback_grad_accum","$FallbackGradAccum","--cache_batch_size","$CacheBatchSize","--device",$Device,"--review_pass_path",$ReviewPassPath,"--max_train_samples","$MaxTrainSamples","--max_test_samples","$MaxTestSamples")
if ($RequireReviewPass) { $args += "--require_review_pass" }
if ($SkipCacheBuild) { $args += "--skip_cache_build" }
& E:\Anaconda\envs\sbw39\python.exe @args
exit $LASTEXITCODE
