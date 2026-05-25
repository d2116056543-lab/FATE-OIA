param(
  [string]$OutputDir = $env:FATE_OIA_OUTPUT_DIR,
  [string]$PretrainedWeights = $env:FATE_OIA_PRETRAINED_WEIGHTS,
  [string]$GroundingCache = $env:FATE_OIA_GROUNDING_CACHE,
  [int]$Epochs = $(if ($env:FATE_OIA_EPOCHS) { [int]$env:FATE_OIA_EPOCHS } else { 30 }),
  [int]$BatchSize = $(if ($env:FATE_OIA_BATCH_SIZE) { [int]$env:FATE_OIA_BATCH_SIZE } else { 1 }),
  [int]$GradAccum = $(if ($env:FATE_OIA_GRAD_ACCUM) { [int]$env:FATE_OIA_GRAD_ACCUM } else { 8 })
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
if (-not $OutputDir) { $OutputDir = ".background_runs\fate_oia_full_360x640" }
if (-not $PretrainedWeights) { $PretrainedWeights = "ckp\reference\dino_deitsmall8_pretrain.pth" }
$Py = if ($env:PYTHON) { $env:PYTHON } else { "E:\Anaconda\envs\sbw39\python.exe" }
$Commit = (git rev-parse HEAD).Trim()

Write-Host "FATE-OIA full-flow template"
Write-Host "git_commit=$Commit"
Write-Host "repo=$Repo"
Write-Host "output_dir=$OutputDir"
Write-Host "pretrained_weights=$PretrainedWeights"
Write-Host "grounding_cache=$GroundingCache"
Write-Host "epochs=$Epochs batch_size=$BatchSize grad_accum=$GradAccum"
Write-Host "compression=keep_merge compression_start_epoch=8 keep_ratio=0.85->0.65"

$cmd = @(
  "-m", "fate_oia.engine.train_fate_oia",
  "--output_dir", $OutputDir,
  "--pretrained_weights", $PretrainedWeights,
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--image_height", "360",
  "--image_width", "640",
  "--preserve_aspect_ratio",
  "--loss", "asl",
  "--loss_action_visual", "0.05",
  "--loss_r2a_gt", "0.10",
  "--loss_action_agree", "0.01",
  "--no-include_fused_branch_loss",
  "--token_compression", "keep_merge",
  "--compression_start_epoch", "8",
  "--compression_warmup_epochs", "6",
  "--compression_keep_ratio_start", "0.85",
  "--compression_keep_ratio_final", "0.65",
  "--num_summary_tokens", "1",
  "--loss_grounding", "0.001",
  "--loss_counterfactual", "0.01",
  "--cf_mask_fill", "mean",
  "--device", "cuda"
)
if ($GroundingCache) {
  $cmd += @("--grounding_cache_jsonl", $GroundingCache)
}
& $Py @cmd
