param([int]$Epochs=20,[int]$BatchSize=8,[int]$GradAccum=4,[int]$FallbackBatchSize=4,[int]$FallbackGradAccum=8,[string]$Device="cuda",[switch]$RequireReviewPass,[string]$OutputDir="")
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
if ($OutputDir -eq "") { $stamp=Get-Date -Format "yyyyMMdd_HHmmss"; $OutputDir=".background_runs\trace_oia_v1_proto_transport_360x640_cache_fulltoken_$stamp" }
$args=@("-m","fate_oia.engine.supervise_trace_oia_foreground","--config","configs\fate_oia_train_360x640_trace_oia_v1.yaml","--output_dir",$OutputDir,"--epochs","$Epochs","--batch_size","$BatchSize","--grad_accum","$GradAccum","--fallback_batch_size","$FallbackBatchSize","--fallback_grad_accum","$FallbackGradAccum","--device",$Device)
if ($RequireReviewPass) { $args += "--require_review_pass" }
& E:\Anaconda\envs\sbw39\python.exe @args
exit $LASTEXITCODE
