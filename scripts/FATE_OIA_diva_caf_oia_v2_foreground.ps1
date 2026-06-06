param(
  [string]$OutputDir = ".background_runs\diva_caf_oia_v2_full",
  [int]$Epochs = 32,
  [int]$BatchSize = 4,
  [int]$Accum = 8
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$review = ".background_runs\diva_caf_oia_v2_preflight\REVIEW_PASS_DIVA_CAF_OIA_V2.txt"
if (!(Test-Path $review)) { throw "Missing REVIEW_PASS_DIVA_CAF_OIA_V2.txt" }
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.train_diva_caf_oia `
  --config configs\fate_oia_train_360x640_diva_caf_oia_v2.yaml `
  --output_dir $OutputDir `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $Accum `
  --device cuda `
  --no_feature_cache `
  --test_only `
  --require_review_pass `
  --print_every 200
