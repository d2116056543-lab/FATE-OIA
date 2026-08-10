param(
    [string]$Worktree = "E:\sbw\FATE_Drive\fate_oia_dual_snapshot_oia_v1_worktree",
    [string]$OutputRoot = ".background_runs\dual_snapshot_oia_v1_full",
    [string]$Python = "E:\Anaconda\envs\sbw39\python.exe",
    [string]$TorchHome = "C:\Users\Lenovo\.cache\torch",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Worktree
$env:PYTHONPATH = $Worktree
$env:PYTHONUNBUFFERED = "1"
$env:TORCH_HOME = $TorchHome
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"

$root = [System.IO.Path]::GetFullPath((Join-Path $Worktree $OutputRoot))
$base = Join-Path $root "base"
$consolidation = Join-Path $root "consolidation"
$ensemble = Join-Path $root "ensemble"
$log = Join-Path $root "full_train.log"
New-Item -ItemType Directory -Force -Path $root | Out-Null

function Invoke-LoggedPython {
    param([string[]]$Arguments)
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $log -Append
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($nativeExitCode -ne 0) {
        throw "Python command failed with exit code ${nativeExitCode}: $($Arguments -join ' ')"
    }
}

if (-not (Test-Path (Join-Path $base "checkpoint_epoch_019.pth"))) {
    $baseArgs = @(
        "-u", "-m", "fate_oia.engine.train_aie_oia",
        "--config", "configs\fate_oia_train_360x640_dual_snapshot_oia_v1.yaml",
        "--output-dir", $base,
        "--run-kind", "full",
        "--epochs", "20", "--batch-size", "6", "--gradient-accumulation-steps", "5",
        "--num-workers", "8", "--device", $Device
    )
    $latest = Join-Path $base "checkpoint_latest.pth"
    if (Test-Path $latest) { $baseArgs += @("--resume", $latest) }
    Invoke-LoggedPython $baseArgs
}

$baseBest = Join-Path $base "checkpoint_best_test_deploy_joint.pth"
if (-not (Test-Path $baseBest)) { throw "Missing base deploy checkpoint: $baseBest" }

if (-not (Test-Path (Join-Path $consolidation "checkpoint_epoch_002.pth"))) {
    $consolidationArgs = @(
        "-u", "-m", "fate_oia.engine.train_aie_oia",
        "--config", "configs\fate_oia_train_360x640_dual_snapshot_oia_v1_consolidation.yaml",
        "--output-dir", $consolidation,
        "--run-kind", "full",
        "--epochs", "3", "--batch-size", "6", "--gradient-accumulation-steps", "5",
        "--num-workers", "8", "--device", $Device
    )
    $latest = Join-Path $consolidation "checkpoint_latest.pth"
    if (Test-Path $latest) {
        $consolidationArgs += @("--resume", $latest)
    } else {
        $consolidationArgs += @("--init-model-checkpoint", $baseBest)
    }
    Invoke-LoggedPython $consolidationArgs
}

$earlyDir = Join-Path $base "epoch_004"
$lateCheckpoint = Join-Path $consolidation "checkpoint_best_test_deploy_joint.pth"
if (-not (Test-Path $lateCheckpoint)) { throw "Missing consolidation deploy checkpoint: $lateCheckpoint" }
$lateEpoch = & $Python -c "import torch; print(int(torch.load(r'$lateCheckpoint', map_location='cpu', weights_only=False)['epoch']))"
if ($LASTEXITCODE -ne 0) { throw "Unable to resolve consolidation snapshot epoch" }
$lateDir = Join-Path $consolidation ("epoch_{0:D3}" -f [int]$lateEpoch)

Invoke-LoggedPython @(
    "-u", "-m", "fate_oia.engine.eval_dual_snapshot_oia",
    "--early-dir", $earlyDir,
    "--late-dir", $lateDir,
    "--output-dir", $ensemble,
    "--action-late-weight", "0.65",
    "--reason-late-weight", "0.875",
    "--action-shrinkage", "50",
    "--reason-shrinkage", "0"
)

$completion = @{
    completed = $true
    completed_at = (Get-Date).ToString("o")
    base_checkpoint = $baseBest
    early_snapshot = $earlyDir
    late_checkpoint = $lateCheckpoint
    late_snapshot = $lateDir
    result = (Join-Path $ensemble "dual_snapshot_result.json")
}
$completion | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $root "GOAL_COMPLETED_DUAL_SNAPSHOT_OIA_V1.json")
