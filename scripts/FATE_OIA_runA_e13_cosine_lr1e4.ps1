param(
  [string]$OutputDir = $env:FATE_OIA_OUTPUT_DIR,
  [string]$Resume = $env:FATE_OIA_RESUME,
  [string]$GroundingCache = $env:FATE_OIA_GROUNDING_CACHE,
  [int]$Epochs = $(if ($env:FATE_OIA_EPOCHS) { [int]$env:FATE_OIA_EPOCHS } else { 24 }),
  [int]$BatchSize = $(if ($env:FATE_OIA_BATCH_SIZE) { [int]$env:FATE_OIA_BATCH_SIZE } else { 4 }),
  [int]$GradAccum = $(if ($env:FATE_OIA_GRAD_ACCUM) { [int]$env:FATE_OIA_GRAD_ACCUM } else { 8 }),
  [switch]$Foreground
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = if ($env:PYTHON) { $env:PYTHON } else { "E:\Anaconda\envs\sbw39\python.exe" }
if (-not $OutputDir) { $OutputDir = ".background_runs\fate_oia_runA_e13_cosine_lr1e4_keep070_cf0_$(Get-Date -Format yyyyMMdd_HHmmss)" }
if (-not $Resume) { $Resume = ".background_runs\fate_oia_full_360x640_dino_20260526_005253\checkpoint_best.pth" }
if (-not $GroundingCache -and (Test-Path ".background_runs\fate_oia_grounding_cache_20260525.jsonl")) { $GroundingCache = ".background_runs\fate_oia_grounding_cache_20260525.jsonl" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$absOut = (Resolve-Path $OutputDir).Path
$trainLog = Join-Path $absOut "train.log"
$errLog = Join-Path $absOut "train.stderr.log"
$launcher = Join-Path $absOut "run_training.cmd"
$cmd = @(
  '-m','fate_oia.engine.train_fate_oia',
  '--config','configs\fate_oia_train_360x640.yaml',
  '--output_dir',$absOut,
  '--resume',$Resume,
  '--no-resume_optimizer',
  '--resume_scheduler',
  '--pretrained_weights','ckp\reference\dino_deitsmall8_pretrain.pth',
  '--pretrained_source','public_dino_reference',
  '--epochs',"$Epochs",
  '--batch_size',"$BatchSize",
  '--gradient_accumulation_steps',"$GradAccum",
  '--num_workers','0',
  '--lr','0.0001',
  '--no-auto_scale_lr',
  '--scheduler','cosine',
  '--min_lr','0.00001',
  '--warmup_epochs','0',
  '--image_height','360',
  '--image_width','640',
  '--preserve_aspect_ratio',
  '--loss','asl',
  '--loss_action_visual','0.05',
  '--loss_r2a_gt','0.10',
  '--loss_action_agree','0.01',
  '--no-include_fused_branch_loss',
  '--loss_grounding','0.0001',
  '--grounding_mode','both',
  '--loss_counterfactual','0',
  '--counterfactual_eval',
  '--counterfactual_start_epoch','0',
  '--counterfactual_topk_ratio','0.05',
  '--cf_mask_fill','mean',
  '--token_compression','keep_merge',
  '--compression_start_epoch','0',
  '--compression_warmup_epochs','0',
  '--compression_keep_ratio_start','0.70',
  '--compression_keep_ratio_final','0.70',
  '--num_summary_tokens','4',
  '--min_tokens','128',
  '--label_correlation','none',
  '--task_balance','none',
  '--best_selection_split','test',
  '--best_selection_metric','joint_test_score',
  '--render_explanation_text',
  '--log_every','1',
  '--device','cuda'
)
if ($GroundingCache) { $cmd += @('--grounding_cache_jsonl', $GroundingCache) }
$manifest = [ordered]@{
  event='fate_oia_runA_launch'; repo=$Repo; output_dir=$absOut; resume=$Resume; epochs=$Epochs;
  batch_size=$BatchSize; gradient_accumulation_steps=$GradAccum; effective_batch_size=($BatchSize*$GradAccum);
  lr='0.0001'; scheduler='cosine'; min_lr='0.00001'; keep_ratio='0.70'; loss_counterfactual='0';
  grounding_cache=$GroundingCache; timestamp=(Get-Date).ToString('s'); command=($Py + ' ' + ($cmd -join ' '))
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $absOut 'launch_manifest.json') -Encoding UTF8
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
  $env:PYTHONUNBUFFERED = '1'
  $env:PYTHONIOENCODING = 'utf-8'
  & $Py @cmd 2>&1 | Tee-Object -FilePath $trainLog
} else {
  $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = 'cmd.exe /c "' + $launcher + '"'
    CurrentDirectory = $Repo
  }
  if ($result.ReturnValue -ne 0) { throw "Win32_Process.Create failed with ReturnValue=$($result.ReturnValue)" }
  Set-Content -LiteralPath (Join-Path $absOut 'train.pid') -Value $result.ProcessId -Encoding ASCII
  Write-Host "FATE-OIA Run A started. LauncherPID=$($result.ProcessId)"
  Write-Host "OutputDir=$absOut"
  Write-Host "TrainLog=$trainLog"
  Write-Host "ErrLog=$errLog"
}
