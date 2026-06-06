param(
  [string]$OutputDir = ".background_runs\diva_caf_oia_v2_full",
  [int]$Epochs = 32,
  [int]$BatchSize = 4,
  [int]$GradAccum = 8,
  [int]$FallbackBatchSize1 = 3,
  [int]$FallbackGradAccum1 = 11,
  [int]$FallbackBatchSize2 = 2,
  [int]$FallbackGradAccum2 = 16,
  [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.supervise_diva_caf_oia_foreground `
  --output_dir $OutputDir `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --grad_accum $GradAccum `
  --fallback_batch_size_1 $FallbackBatchSize1 `
  --fallback_grad_accum_1 $FallbackGradAccum1 `
  --fallback_batch_size_2 $FallbackBatchSize2 `
  --fallback_grad_accum_2 $FallbackGradAccum2 `
  --device $Device
