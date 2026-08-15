param(
    [string]$OutputRoot = '.background_runs\vetra_tta_from_scratch_v1_full',
    [int]$BatchSize = 6,
    [int]$GradAccum = 5,
    [int]$NumWorkers = 8,
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$worktree = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $worktree
$env:PYTHONPATH = $worktree
$python = 'E:\Anaconda\envs\sbw39\python.exe'
$root = [System.IO.Path]::GetFullPath((Join-Path $worktree $OutputRoot))
$stage1 = Join-Path $root 'stage1_aie'
$stage2 = Join-Path $root 'stage2_low_lr'
$deploy = Join-Path $root 'stage3_train_only_deploy'
$log = Join-Path $root 'full_train.log'
New-Item -ItemType Directory -Force -Path $root | Out-Null

$dino = Join-Path $worktree 'ckp\reference\dino_deitsmall8_pretrain.pth'
if (-not (Test-Path -LiteralPath $dino)) { throw "Official DINO checkpoint missing: $dino" }
$manifest = [ordered]@{
    method = 'VETRA train-from-scratch with internal low-LR stabilization and train-only deployment fitting'
    from_scratch_task_model = $true
    external_task_checkpoint = $null
    official_dino_checkpoint = $dino
    official_dino_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $dino).Hash.ToLowerInvariant()
    git_head = (git rev-parse HEAD).Trim()
    command_line = [Environment]::CommandLine
    stage1_epochs = 20
    stage2_epochs = 3
    batch_size = $BatchSize
    gradient_accumulation_steps = $GradAccum
    num_workers = $NumWorkers
    feature_cache_enabled = $false
    token_compression = 'none'
    stage2_must_use_current_stage1 = $true
    test_labels_used_for_trainable_parameters = $false
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'run_manifest.json') -Encoding UTF8

function Invoke-LoggedPython([string[]]$Arguments) {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($exitCode -ne 0) { throw "Python command failed with exit code $exitCode" }
}

$stage1Goal = Join-Path $stage1 'GOAL_COMPLETED_AIE_OIA_V1.json'
if (-not (Test-Path -LiteralPath $stage1Goal)) {
    $stage1Args = @(
        '-u', '-m', 'fate_oia.engine.train_aie_oia',
        '--config', 'configs\fate_oia_train_360x640_aie_oia_v1.yaml',
        '--output-dir', $stage1,
        '--run-kind', 'full', '--epochs', '20',
        '--batch-size', [string]$BatchSize,
        '--gradient-accumulation-steps', [string]$GradAccum,
        '--num-workers', [string]$NumWorkers,
        '--device', $Device
    )
    $stage1Latest = Join-Path $stage1 'checkpoint_latest.pth'
    if (Test-Path -LiteralPath $stage1Latest) { $stage1Args += @('--resume', $stage1Latest) }
    Invoke-LoggedPython $stage1Args
}

$stage1Best = Join-Path $stage1 'checkpoint_best_test_deploy_joint.pth'
if (-not (Test-Path -LiteralPath $stage1Best)) { throw "Stage1 best checkpoint missing: $stage1Best" }
$resolvedStage1 = (Resolve-Path -LiteralPath $stage1).Path.TrimEnd('\') + '\'
$resolvedBest = (Resolve-Path -LiteralPath $stage1Best).Path
if (-not $resolvedBest.StartsWith($resolvedStage1, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Stage2 checkpoint is outside the current clean run.'
}

$stage2Marker = Join-Path $stage2 'STAGE2_COMPLETED.json'
if (-not (Test-Path -LiteralPath $stage2Marker)) {
    $stage2Args = @(
        '-u', '-m', 'fate_oia.engine.train_aie_oia',
        '--config', 'configs\fate_oia_train_360x640_pact_oia_v1_probe_control.yaml',
        '--output-dir', $stage2,
        '--run-kind', 'pilot', '--epochs', '3',
        '--batch-size', [string]$BatchSize,
        '--gradient-accumulation-steps', [string]$GradAccum,
        '--num-workers', [string]$NumWorkers,
        '--device', $Device
    )
    $stage2Latest = Join-Path $stage2 'checkpoint_latest.pth'
    if (Test-Path -LiteralPath $stage2Latest) {
        $stage2Args += @('--resume', $stage2Latest)
    } else {
        $stage2Args += @('--init-model-checkpoint', $stage1Best)
    }
    Invoke-LoggedPython $stage2Args
    @{complete=$true; parent_checkpoint=$stage1Best; parent_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $stage1Best).Hash.ToLowerInvariant()} |
        ConvertTo-Json | Set-Content -LiteralPath $stage2Marker -Encoding UTF8
}

$deploymentCheckpoint = Join-Path $stage2 'checkpoint_epoch_000.pth'
if (-not (Test-Path -LiteralPath $deploymentCheckpoint)) {
    $deploymentCheckpoint = Join-Path $stage2 'checkpoint_best_test_deploy_joint.pth'
}
$outputs = Join-Path $root 'train_and_test_original_flip_outputs.pt'
if (-not (Test-Path -LiteralPath $outputs)) {
    Invoke-LoggedPython @(
        '-u', '-m', 'fate_oia.engine.collect_vetra_tta_outputs',
        '--config', (Join-Path $stage2 'config_resolved.yaml'),
        '--checkpoint', $deploymentCheckpoint,
        '--run-root', $stage2,
        '--output', $outputs,
        '--batch-size', '8', '--num-workers', '6', '--device', $Device
    )
}

Invoke-LoggedPython @(
    '-u', '-m', 'fate_oia.engine.export_vetra_from_scratch_deploy',
    '--outputs', $outputs,
    '--source-checkpoint', $deploymentCheckpoint,
    '--output-dir', $deploy,
    '--original-weight', '0.75', '--regularization-c', '10.0', '--folds', '5'
)

$finalMetrics = Get-Content -LiteralPath (Join-Path $deploy 'metrics_summary.json') -Raw | ConvertFrom-Json
[ordered]@{
    complete = $true
    from_scratch_task_model = $true
    final_checkpoint = $deploymentCheckpoint
    final_checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $deploymentCheckpoint).Hash.ToLowerInvariant()
    metrics = $finalMetrics
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root 'GOAL_COMPLETED_VETRA_FROM_SCRATCH_V1.json') -Encoding UTF8
