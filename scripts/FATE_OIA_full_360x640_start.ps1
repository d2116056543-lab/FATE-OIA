param(
  [string]$OutputDir = $env:FATE_OIA_OUTPUT_DIR,
  [string]$PretrainedWeights = $env:FATE_OIA_PRETRAINED_WEIGHTS,
  [string]$GroundingCache = $env:FATE_OIA_GROUNDING_CACHE,
  [int]$Epochs = $(if ($env:FATE_OIA_EPOCHS) { [int]$env:FATE_OIA_EPOCHS } else { 40 }),
  [int]$BatchSize = $(if ($env:FATE_OIA_BATCH_SIZE) { [int]$env:FATE_OIA_BATCH_SIZE } else { 1 }),
  [int]$GradAccum = $(if ($env:FATE_OIA_GRAD_ACCUM) { [int]$env:FATE_OIA_GRAD_ACCUM } else { 32 }),
  [int]$NumWorkers = $(if ($env:FATE_OIA_NUM_WORKERS) { [int]$env:FATE_OIA_NUM_WORKERS } else { 0 }),
  [switch]$Foreground
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = if ($env:PYTHON) { $env:PYTHON } else { "E:\Anaconda\envs\sbw39\python.exe" }
if (-not $OutputDir) { $OutputDir = ".background_runs\fate_oia_full_360x640_$(Get-Date -Format yyyyMMdd_HHmmss)" }
if (-not $PretrainedWeights) { $PretrainedWeights = "ckp\reference\dino_deitsmall8_pretrain.pth" }
if (-not $GroundingCache -and (Test-Path ".background_runs\fate_oia_grounding_cache_20260525.jsonl")) { $GroundingCache = ".background_runs\fate_oia_grounding_cache_20260525.jsonl" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$absOut = (Resolve-Path $OutputDir).Path
$trainLog = Join-Path $absOut "train.log"
$errLog = Join-Path $absOut "train.stderr.log"
$pidFile = Join-Path $absOut "train.pid"
$launcher = Join-Path $absOut "run_training.ps1"
$cmd = @(
  '-m','fate_oia.engine.train_fate_oia',
  '--config','configs\fate_oia_train_360x640.yaml',
  '--output_dir',$absOut,
  '--pretrained_weights',$PretrainedWeights,
  '--pretrained_source','public_dino_reference',
  '--epochs',"$Epochs",
  '--batch_size',"$BatchSize",
  '--gradient_accumulation_steps',"$GradAccum",
  '--num_workers',"$NumWorkers",
  '--auto_scale_lr',
  '--reference_effective_batch','32',
  '--base_head_lr_at_reference_batch','0.0003',
  '--max_head_lr','0.0005',
  '--image_height','360',
  '--image_width','640',
  '--preserve_aspect_ratio',
  '--loss','asl',
  '--loss_action_visual','0.05',
  '--loss_r2a_gt','0.10',
  '--loss_action_agree','0.01',
  '--no-include_fused_branch_loss',
  '--loss_grounding','0.0003',
  '--grounding_mode','both',
  '--loss_counterfactual','0.005',
  '--counterfactual_start_epoch','8',
  '--counterfactual_topk_ratio','0.05',
  '--cf_mask_fill','mean',
  '--token_compression','keep_merge',
  '--compression_start_epoch','8',
  '--compression_warmup_epochs','6',
  '--compression_keep_ratio_start','0.85',
  '--compression_keep_ratio_final','0.65',
  '--num_summary_tokens','4',
  '--min_tokens','128',
  '--best_selection_split','test',
  '--best_selection_metric','joint_test_score',
  '--render_explanation_text',
  '--log_every','1',
  '--device','cuda'
)
if ($GroundingCache) { $cmd += @('--grounding_cache_jsonl', $GroundingCache) }
$manifest = [ordered]@{
  event='fate_oia_launch'; repo=$Repo; output_dir=$absOut; pretrained_weights=$PretrainedWeights;
  pretrained_source='public_dino_reference'; using_classifier_head=$false; epochs=$Epochs; batch_size=$BatchSize;
  gradient_accumulation_steps=$GradAccum; effective_batch_size=($BatchSize*$GradAccum); grounding_cache=$GroundingCache;
  best_selection_split='test'; best_selection_metric='joint_test_score'; command=($Py + ' ' + ($cmd -join ' ')); timestamp=(Get-Date).ToString('s')
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $absOut 'launch_manifest.json') -Encoding UTF8
$quoted = ($cmd | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join " "
Set-Content -LiteralPath $launcher -Encoding UTF8 -Value @"
`$ErrorActionPreference = 'Stop'
Set-Location '$Repo'
`$env:PYTHONUNBUFFERED = '1'
`$env:PYTHONIOENCODING = 'utf-8'
& '$Py' $quoted
"@
if ($Foreground) {
  $env:PYTHONUNBUFFERED = '1'
  $env:PYTHONIOENCODING = 'utf-8'
  & $Py @cmd 2>&1 | Tee-Object -FilePath $trainLog
} else {
  $envBlock = @(
    'PYTHONUNBUFFERED=1',
    'PYTHONIOENCODING=utf-8'
  )
  # Start python directly. A nested powershell process can hand Python an invalid
  # redirected stdout handle on this Windows/OpenSSH setup.
  $p = Start-Process -FilePath $Py -ArgumentList $cmd -WorkingDirectory $Repo -RedirectStandardOutput $trainLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
  Set-Content -LiteralPath $pidFile -Value $p.Id -Encoding ASCII
  Write-Host "FATE-OIA training started. PID=$($p.Id)"
  Write-Host "OutputDir=$absOut"
  Write-Host "TrainLog=$trainLog"
  Write-Host "ErrLog=$errLog"
}
