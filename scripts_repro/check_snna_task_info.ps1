param(
    [string]$TaskName = 'SNNA_full_repro_current',
    [string]$RunDir = 'E:\sbw\SNNA_repro\SNNA\repro_runs\full_20260522_183454'
)
$ErrorActionPreference = 'SilentlyContinue'
Write-Host "---TASK INFO---"
Get-ScheduledTask -TaskName $TaskName | Format-List TaskName,State
Get-ScheduledTaskInfo -TaskName $TaskName | Format-List LastRunTime,LastTaskResult,NextRunTime
Write-Host "---RUN FILES---"
Get-ChildItem $RunDir -Recurse | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
Write-Host "---EXIT CODE---"
if (Test-Path "$RunDir\task_exit_code.txt") { Get-Content "$RunDir\task_exit_code.txt" -Raw }
Write-Host "---TASK EXCEPTION---"
if (Test-Path "$RunDir\logs\task_exception.log") { Get-Content "$RunDir\logs\task_exception.log" -Raw }
