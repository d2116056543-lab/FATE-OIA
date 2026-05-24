param(
    [string]$RunDir = ''
)

$ErrorActionPreference = 'SilentlyContinue'
$repo = 'E:\sbw\SNNA_repro\SNNA'
if (-not $RunDir) {
    if (Test-Path "$repo\repro_runs\LATEST_SNNA_FULL.txt") {
        $RunDir = (Get-Content "$repo\repro_runs\LATEST_SNNA_FULL.txt" -Raw).Trim()
    }
}

$log = Join-Path $RunDir 'logs\dino_full.log'
$vram = Join-Path $RunDir 'logs\vram_monitor.csv'
Write-Host "---RUN---"
Write-Host $RunDir
Write-Host "---TASK---"
Get-ScheduledTask -TaskName 'SNNA_full_repro_current' | Format-List TaskName,State
Get-ScheduledTaskInfo -TaskName 'SNNA_full_repro_current' | Format-List LastRunTime,LastTaskResult
Write-Host "---LATEST DINO PROGRESS---"
if (Test-Path $log) {
    $lines = Select-String -Path $log -Pattern 'Epoch: \[[0-9]+/200\].*\[[ ]*[0-9]+/12500\]' | ForEach-Object { $_.Line }
    $tail = $lines | Select-Object -Last 12
    $tail
    $last = $lines | Select-Object -Last 1
    if ($last -match 'Epoch: \[(\d+)/200\]\s+\[\s*(\d+)/12500\].*time:\s*([0-9.]+)') {
        $epoch = [int]$Matches[1]
        $iter = [int]$Matches[2]
        $sec = [double]$Matches[3]
        $epochHours = 12500.0 * $sec / 3600.0
        $totalDays = 200.0 * $epochHours / 24.0
        $doneFrac = ($epoch * 12500.0 + $iter) / (200.0 * 12500.0)
        Write-Host "---SPEED ESTIMATE FROM LAST LOGGED ITER---"
        Write-Host ("last_epoch={0} last_iter={1} iter_time_sec={2:n3}" -f $epoch,$iter,$sec)
        Write-Host ("estimated_hours_per_epoch={0:n2}" -f $epochHours)
        Write-Host ("estimated_days_for_200_epochs={0:n2}" -f $totalDays)
        Write-Host ("logged_progress_percent={0:n3}" -f ($doneFrac*100))
    }
}
Write-Host "---VRAM/GPU LAST---"
if (Test-Path $vram) {
    Get-Content $vram -Tail 12
    $rows = Get-Content $vram | Select-Object -Skip 1
    $maxMem = 0
    $sumUtil = 0
    $cnt = 0
    foreach ($r in $rows) {
        $c = $r -split ','
        if ($c.Count -ge 4) {
            $mem = 0
            $util = 0
            [void][int]::TryParse($c[1].Trim(), [ref]$mem)
            [void][int]::TryParse($c[3].Trim(), [ref]$util)
            if ($mem -gt $maxMem) { $maxMem = $mem }
            $sumUtil += $util
            $cnt += 1
        }
    }
    if ($cnt -gt 0) {
        Write-Host "---GPU SUMMARY---"
        Write-Host ("max_mem_mib={0} avg_util_percent={1:n1}" -f $maxMem, ($sumUtil/$cnt))
    }
}
