param(
  [string]$RunName = $(if ($env:FATE_OIA_RUN_NAME) { $env:FATE_OIA_RUN_NAME } else { "controlled_resume" }),
  [string]$OutputDir = $env:FATE_OIA_OUTPUT_DIR,
  [string]$Resume = $env:FATE_OIA_RESUME,
  [string]$GroundingCache = $env:FATE_OIA_GROUNDING_CACHE,
  [int]$Epochs = $(if ($env:FATE_OIA_EPOCHS) { [int]$env:FATE_OIA_EPOCHS } else { 24 }),
  [int]$BatchSize = $(if ($env:FATE_OIA_BATCH_SIZE) { [int]$env:FATE_OIA_BATCH_SIZE } else { 4 }),
  [int]$GradAccum = $(if ($env:FATE_OIA_GRAD_ACCUM) { [int]$env:FATE_OIA_GRAD_ACCUM } else { 8 }),
  [string]$Lr = $(if ($env:FATE_OIA_LR) { $env:FATE_OIA_LR } else { "0.0001" }),
  [ValidateSet("none", "cosine", "plateau")]
  [string]$Scheduler = $(if ($env:FATE_OIA_SCHEDULER) { $env:FATE_OIA_SCHEDULER } else { "cosine" }),
  [string]$MinLr = $(if ($env:FATE_OIA_MIN_LR) { $env:FATE_OIA_MIN_LR } else { "0.00001" }),
  [string]$KeepRatio = $(if ($env:FATE_OIA_KEEP_RATIO) { $env:FATE_OIA_KEEP_RATIO } else { "0.70" }),
  [string]$LossGrounding = $(if ($env:FATE_OIA_LOSS_GROUNDING) { $env:FATE_OIA_LOSS_GROUNDING } else { "0.0001" }),
  [string]$LossCounterfactual = $(if ($env:FATE_OIA_LOSS_COUNTERFACTUAL) { $env:FATE_OIA_LOSS_COUNTERFACTUAL } else { "0" }),
  [ValidateSet("none", "self_attn")]
  [string]$LabelCorrelation = $(if ($env:FATE_OIA_LABEL_CORRELATION) { $env:FATE_OIA_LABEL_CORRELATION } else { "none" }),
  [ValidateSet("none", "uncertainty")]
  [string]$TaskBalance = $(if ($env:FATE_OIA_TASK_BALANCE) { $env:FATE_OIA_TASK_BALANCE } else { "none" }),
  [switch]$ResumeStrict,
  [switch]$AllowConcurrent,
  [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = if ($env:PYTHON) { $env:PYTHON } else { "E:\Anaconda\envs\sbw39\python.exe" }
if (-not $OutputDir) { $OutputDir = ".background_runs\fate_oia_${RunName}_$(Get-Date -Format yyyyMMdd_HHmmss)" }
if (-not $Resume) { $Resume = ".background_runs\fate_oia_full_360x640_dino_20260526_005253\checkpoint_best.pth" }
if (-not $GroundingCache -and (Test-Path ".background_runs\fate_oia_grounding_cache_20260525.jsonl")) {
  $GroundingCache = ".background_runs\fate_oia_grounding_cache_20260525.jsonl"
}

if (-not $AllowConcurrent) {
  $active = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'fate_oia\.engine\.train_fate_oia' -and $_.CommandLine -notmatch [regex]::Escape($OutputDir)
  }
  if ($active) {
    $pids = ($active | ForEach-Object { "$($_.ProcessId):$($_.Name)" }) -join ", "
    throw "Another FATE-OIA training process is active ($pids). Stop it or pass -AllowConcurrent intentionally."
  }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$absOut = (Resolve-Path $OutputDir).Path
$trainLog = Join-Path $absOut "train.log"
$errLog = Join-Path $absOut "train.stderr.log"
$launcher = Join-Path $absOut "run_training.cmd"
$strictFlag = if ($ResumeStrict -or $LabelCorrelation -eq "none") { "--resume_strict" } else { "--no-resume_strict" }

$cmd = @(
  "-m", "fate_oia.engine.train_fate_oia",
  "--config", "configs\fate_oia_train_360x640.yaml",
  "--output_dir", $absOut,
  "--resume", $Resume,
  "--no-resume_optimizer",
  "--resume_scheduler",
  $strictFlag,
  "--pretrained_weights", "ckp\reference\dino_deitsmall8_pretrain.pth",
  "--pretrained_source", "public_dino_reference",
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--num_workers", "0",
  "--lr", $Lr,
  "--no-auto_scale_lr",
  "--scheduler", $Scheduler,
  "--min_lr", $MinLr,
  "--warmup_epochs", "0",
  "--image_height", "360",
  "--image_width", "640",
  "--preserve_aspect_ratio",
  "--loss", "asl",
  "--loss_action_visual", "0.05",
  "--loss_r2a_gt", "0.10",
  "--loss_action_agree", "0.01",
  "--no-include_fused_branch_loss",
  "--loss_grounding", $LossGrounding,
  "--grounding_mode", "both",
  "--loss_counterfactual", $LossCounterfactual,
  "--counterfactual_eval",
  "--counterfactual_start_epoch", "0",
  "--counterfactual_topk_ratio", "0.05",
  "--cf_mask_fill", "mean",
  "--token_compression", "keep_merge",
  "--compression_start_epoch", "0",
  "--compression_warmup_epochs", "0",
  "--compression_keep_ratio_start", $KeepRatio,
  "--compression_keep_ratio_final", $KeepRatio,
  "--num_summary_tokens", "4",
  "--min_tokens", "128",
  "--label_correlation", $LabelCorrelation,
  "--label_correlation_layers", "1",
  "--label_correlation_heads", "4",
  "--label_correlation_bias", "none",
  "--task_balance", $TaskBalance,
  "--best_selection_split", "test",
  "--best_selection_metric", "joint_test_score",
  "--render_explanation_text",
  "--log_every", "1",
  "--device", "cuda"
)
if ($GroundingCache) { $cmd += @("--grounding_cache_jsonl", $GroundingCache) }

$manifest = [ordered]@{
  event = "fate_oia_controlled_resume_launch"
  run_name = $RunName
  repo = $Repo
  git_commit = (git rev-parse HEAD).Trim()
  output_dir = $absOut
  resume = $Resume
  epochs = $Epochs
  batch_size = $BatchSize
  gradient_accumulation_steps = $GradAccum
  effective_batch_size = ($BatchSize * $GradAccum)
  lr = $Lr
  scheduler = $Scheduler
  min_lr = $MinLr
  keep_ratio = $KeepRatio
  loss_grounding = $LossGrounding
  loss_counterfactual = $LossCounterfactual
  label_correlation = $LabelCorrelation
  task_balance = $TaskBalance
  resume_strict = ($strictFlag -eq "--resume_strict")
  grounding_cache = $GroundingCache
  timestamp = (Get-Date).ToString("s")
  command = ($Py + " " + ($cmd -join " "))
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $absOut "launch_manifest.json") -Encoding UTF8
$cmdQuoted = ($cmd | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join " "
Set-Content -LiteralPath $launcher -Encoding ASCII -Value @"
@echo off
cd /d "$Repo"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
"$Py" $cmdQuoted > "$trainLog" 2> "$errLog"
exit /b %ERRORLEVEL%
"@

if ($Foreground) {
  & $Py @cmd
} else {
  $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$launcher`"" -PassThru -WindowStyle Hidden
  $proc.Id | Set-Content -LiteralPath (Join-Path $absOut "train.pid") -Encoding ASCII
  Write-Host "Started FATE-OIA controlled run."
  Write-Host "RunName=$RunName"
  Write-Host "LauncherPID=$($proc.Id)"
  Write-Host "OutputDir=$absOut"
  Write-Host "TrainLog=$trainLog"
  Write-Host "ErrLog=$errLog"
}
