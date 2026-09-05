$ErrorActionPreference = "Stop"

$taskName = "TIDA_V46_ACTION_TOKEN_REASON_PILOT4096"
$script = "E:\sbw\FATE_Drive\fate_oia_tida_site_v20_worktree\scripts\run_v46_pilot4096.ps1"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 4) -StartWhenAvailable
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -User $env:USERNAME -RunLevel Highest | Out-Null
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName
"TASK|$($task.State)|last=$($info.LastRunTime.ToString('s'))|code=$($info.LastTaskResult)"
