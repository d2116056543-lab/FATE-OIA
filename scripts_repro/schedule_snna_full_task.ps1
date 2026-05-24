param(
    [int]$DinoBatch = 8,
    [int]$ClassifierBatch = 4,
    [string]$TaskName = 'SNNA_full_repro_current'
)

$ErrorActionPreference = 'Stop'
$script = 'E:\sbw\SNNA_repro\SNNA\scripts_repro\run_snna_full_task.ps1'

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$args = "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -DinoBatch $DinoBatch -ClassifierBatch $ClassifierBatch"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $args
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $TaskName

$latestPath = 'E:\sbw\SNNA_repro\SNNA\repro_runs\LATEST_SNNA_FULL.txt'
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Path $latestPath) {
        $runDir = Get-Content $latestPath -Raw
        if ($runDir.Trim()) { break }
    }
    Start-Sleep -Seconds 1
}

[ordered]@{
    task_name = $TaskName
    dino_batch = $DinoBatch
    classifier_batch = $ClassifierBatch
    latest_run_file = $latestPath
    latest_run_dir = if (Test-Path $latestPath) { (Get-Content $latestPath -Raw).Trim() } else { '' }
} | ConvertTo-Json -Depth 4
