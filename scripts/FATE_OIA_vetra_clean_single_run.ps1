param(
    [string]$OutputRoot = 'F:\FATE_Drive_runs\vetra_clean_single_run',
    [int]$Epochs = 20,
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
$root = [System.IO.Path]::GetFullPath($OutputRoot)
$modelRun = Join-Path $root 'model_training'
$deployment = Join-Path $root 'train_only_deployment'
$outputs = Join-Path $root 'train_calib_audit_test_outputs.pt'
$log = Join-Path $root 'full_train.log'
New-Item -ItemType Directory -Force -Path $root | Out-Null

$dino = Join-Path $worktree 'ckp\reference\dino_deitsmall8_pretrain.pth'
if (-not (Test-Path -LiteralPath $dino)) {
    $sharedDino = 'E:\sbw\FATE_Drive\fate_oia_worktree\ckp\reference\dino_deitsmall8_pretrain.pth'
    if (-not (Test-Path -LiteralPath $sharedDino)) {
        throw "Official DINO checkpoint missing from worktree and shared reference: $dino"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dino) | Out-Null
    New-Item -ItemType HardLink -Path $dino -Target $sharedDino | Out-Null
}

function Invoke-LoggedPython([string[]]$Arguments) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($exitCode -ne 0) { throw "Python command failed with exit code $exitCode" }
}

$manifest = [ordered]@{
    method = 'single clean direct-image VETRA training with train-only deployment fitting'
    git_head = (git rev-parse HEAD).Trim()
    official_dino_checkpoint = $dino
    official_dino_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $dino).Hash.ToLowerInvariant()
    random_task_head_initialization = $true
    external_task_checkpoint = $null
    checkpoint_selection_split = 'train_audit'
    deployment_fit_split = 'train_calib'
    test_labels_used_for_parameters = $false
    feature_cache_enabled = $false
    token_compression = 'none'
    epochs = $Epochs
    batch_size = $BatchSize
    gradient_accumulation_steps = $GradAccum
    num_workers = $NumWorkers
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root 'run_manifest.json') -Encoding UTF8

$goal = Join-Path $modelRun 'GOAL_COMPLETED_AIE_OIA_V1.json'
if (-not (Test-Path -LiteralPath $goal)) {
    $arguments = @(
        '-u', '-m', 'fate_oia.engine.train_aie_oia',
        '--config', 'configs\fate_oia_train_360x640_vetra_clean_single_run.yaml',
        '--output-dir', $modelRun,
        '--run-kind', 'full',
        '--epochs', [string]$Epochs,
        '--batch-size', [string]$BatchSize,
        '--gradient-accumulation-steps', [string]$GradAccum,
        '--num-workers', [string]$NumWorkers,
        '--device', $Device
    )
    $latest = Join-Path $modelRun 'checkpoint_latest.pth'
    if (Test-Path -LiteralPath $latest) { $arguments += @('--resume', $latest) }
    Invoke-LoggedPython $arguments
}

$selected = Join-Path $modelRun 'checkpoint_final_train_audit_selected.pth'
if (-not (Test-Path -LiteralPath $selected)) {
    throw "Train-audit-selected checkpoint missing: $selected"
}
if (-not (Test-Path -LiteralPath $outputs)) {
    Invoke-LoggedPython @(
        '-u', '-m', 'fate_oia.engine.collect_vetra_tta_outputs',
        '--config', (Join-Path $modelRun 'config_resolved.yaml'),
        '--checkpoint', $selected,
        '--run-root', $modelRun,
        '--output', $outputs,
        '--batch-size', '8',
        '--num-workers', '8',
        '--device', $Device
    )
}

Invoke-LoggedPython @(
    '-u', '-m', 'fate_oia.engine.export_vetra_from_scratch_deploy',
    '--outputs', $outputs,
    '--source-checkpoint', $selected,
    '--output-dir', $deployment,
    '--fit-splits', 'train_calib',
    '--original-weight', '0.75',
    '--regularization-c', '10.0',
    '--folds', '5'
)

$metrics = Get-Content -LiteralPath (Join-Path $deployment 'metrics_summary.json') -Raw | ConvertFrom-Json
[ordered]@{
    complete = $true
    single_clean_run = $true
    model_checkpoint = $selected
    model_checkpoint_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $selected).Hash.ToLowerInvariant()
    deployment_checkpoint = (Join-Path $deployment 'vetra_from_scratch_deploy.pth')
    metrics = $metrics
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root 'GOAL_COMPLETED_VETRA_CLEAN_SINGLE_RUN.json') -Encoding UTF8
