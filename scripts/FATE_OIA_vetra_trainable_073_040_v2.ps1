param(
    [string]$OutputRoot = '.background_runs\vetra_trainable_073_040_v2_full',
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
$joint = Join-Path $root 'stage1_joint'
$refine = Join-Path $root 'stage2_reason_refine'
New-Item -ItemType Directory -Force -Path $root | Out-Null

function Invoke-Training([string[]]$Arguments) {
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path (Join-Path $joint 'GOAL_COMPLETED_AIE_OIA_V1.json'))) {
    Invoke-Training @(
        '-u', '-m', 'fate_oia.engine.train_aie_oia',
        '--config', 'configs\fate_oia_train_360x640_vetra_trainable_073_040_v2_joint.yaml',
        '--output-dir', $joint, '--run-kind', 'full', '--epochs', '14',
        '--batch-size', [string]$BatchSize,
        '--gradient-accumulation-steps', [string]$GradAccum,
        '--num-workers', [string]$NumWorkers, '--device', $Device
    )
}

$parent = Join-Path $joint 'checkpoint_best_test_deploy_joint.pth'
if (-not (Test-Path $parent)) { throw "Joint-stage best checkpoint missing: $parent" }
if (-not (Test-Path (Join-Path $refine 'GOAL_COMPLETED_AIE_OIA_V1.json'))) {
    Invoke-Training @(
        '-u', '-m', 'fate_oia.engine.train_aie_oia',
        '--config', 'configs\fate_oia_train_360x640_vetra_trainable_073_040_v2_reason_refine.yaml',
        '--output-dir', $refine, '--run-kind', 'full', '--epochs', '3',
        '--batch-size', [string]$BatchSize,
        '--gradient-accumulation-steps', [string]$GradAccum,
        '--num-workers', [string]$NumWorkers, '--device', $Device,
        '--init-model-checkpoint', $parent
    )
}

$jointGoal = Get-Content (Join-Path $joint 'GOAL_COMPLETED_AIE_OIA_V1.json') -Raw | ConvertFrom-Json
$refineGoal = Get-Content (Join-Path $refine 'GOAL_COMPLETED_AIE_OIA_V1.json') -Raw | ConvertFrom-Json
[ordered]@{
    complete = $true
    from_scratch_task_model = $true
    external_task_checkpoint = $null
    joint_best = $jointGoal.best
    reason_refine_best = $refineGoal.best
} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $root 'GOAL_COMPLETED_VETRA_TRAINABLE_073_040_V2.json') -Encoding UTF8
