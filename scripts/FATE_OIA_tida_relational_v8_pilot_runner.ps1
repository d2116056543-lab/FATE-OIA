param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Worktree,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [int]$Epochs = 1,
    [int]$MaxSamples = 1024,
    [int]$MaxEvalSamples = 512
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stdout = Join-Path $OutputDir 'pilot_stdout.log'
$stderr = Join-Path $OutputDir 'pilot_stderr.log'
$exitCode = Join-Path $OutputDir 'process_exit_code.txt'
$completed = Join-Path $OutputDir 'runner_completed.json'
Remove-Item -LiteralPath $exitCode, $completed -Force -ErrorAction SilentlyContinue

$arguments = @(
    '-u', '-m', 'fate_oia.engine.train_tida_oia',
    '--config', 'configs\fate_oia_train_tida_relational_flow_v8_10k.yaml',
    '--clip-manifest', 'artifacts\tida_10k_v8\tida_10k_primary_manifest.jsonl',
    '--image-checkpoint', 'F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth',
    '--checkpoint', $Checkpoint,
    '--output-dir', $OutputDir,
    '--epochs', [string]$Epochs,
    '--batch-size', '6',
    '--gradient-accumulation-steps', '1',
    '--context-chunk-size', '2',
    '--num-workers', '6',
    '--run-kind', 'smoke',
    '--max-samples', [string]$MaxSamples,
    '--max-eval-samples', [string]$MaxEvalSamples,
    '--schedule-total-updates', [string]([math]::Ceiling($MaxSamples / 6) * $Epochs),
    '--skip-ema-eval',
    '--skip-expanded-eval',
    '--train-owners', 'relational_traffic_action,relational_traffic_reason',
    '--device', 'cuda'
)

try {
    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $arguments `
        -WorkingDirectory $Worktree `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -NoNewWindow `
        -Wait `
        -PassThru
    $code = $process.ExitCode
} catch {
    $_ | Out-String | Add-Content -LiteralPath $stderr -Encoding UTF8
    $code = 1
}

Set-Content -LiteralPath $exitCode -Value $code -Encoding ASCII
@{
    exit_code = $code
    completed_at = (Get-Date).ToString('o')
    output_dir = $OutputDir
} | ConvertTo-Json | Set-Content -LiteralPath $completed -Encoding UTF8
exit $code
