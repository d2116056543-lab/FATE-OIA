param(
    [Parameter(Mandatory = $true)][int]$PythonPid,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$statusPath = Join-Path $OutputDirectory "supervisor_status.json"
$heartbeatPath = Join-Path $OutputDirectory "supervisor_heartbeat.jsonl"
$startedAt = (Get-Date).ToString("o")

function Write-WatchStatus {
    param([string]$State, [bool]$Alive)
    $epochDirs = @(Get-ChildItem -LiteralPath $OutputDirectory -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^epoch_[0-9]+$' } |
        Sort-Object Name)
    $latest = if ($epochDirs.Count -gt 0) { $epochDirs[-1].Name } else { $null }
    $metrics = @(Get-ChildItem -LiteralPath $OutputDirectory -Recurse -Filter 'metrics_summary.json' -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending)
    $loss = @(Get-ChildItem -LiteralPath $OutputDirectory -Recurse -Filter 'loss_components.jsonl' -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending)
    $payload = [ordered]@{
        state = $State
        python_pid = $PythonPid
        python_alive = $Alive
        watcher_pid = $PID
        started_at = $startedAt
        updated_at = (Get-Date).ToString("o")
        latest_epoch_dir = $latest
        latest_metrics_path = if ($metrics.Count -gt 0) { $metrics[0].FullName } else { $null }
        latest_metrics_mtime = if ($metrics.Count -gt 0) { $metrics[0].LastWriteTime.ToString("o") } else { $null }
        latest_loss_path = if ($loss.Count -gt 0) { $loss[0].FullName } else { $null }
        latest_loss_mtime = if ($loss.Count -gt 0) { $loss[0].LastWriteTime.ToString("o") } else { $null }
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
    $payload | ConvertTo-Json -Compress -Depth 5 | Add-Content -LiteralPath $heartbeatPath -Encoding UTF8
}

while ($true) {
    $proc = Get-Process -Id $PythonPid -ErrorAction SilentlyContinue
    $alive = $null -ne $proc
    if ($alive) {
        Write-WatchStatus -State "running" -Alive $true
        Start-Sleep -Seconds 15
        continue
    }
    $goal = Test-Path (Join-Path $OutputDirectory "GOAL_COMPLETED_MOSAIC_TRUST_V5_CREDO_MAP.json")
    Write-WatchStatus -State $(if ($goal) { "completed" } else { "python_exited" }) -Alive $false
    break
}
