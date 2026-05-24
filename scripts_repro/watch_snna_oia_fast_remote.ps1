param(
    [string]$RunDir = ''
)

$ErrorActionPreference = 'SilentlyContinue'
$repo = 'E:\sbw\SNNA_repro\SNNA'
Set-Location $repo
if (-not $RunDir) {
    $latestPath = "$repo\repro_runs\LATEST_SNNA_OIA_FAST.txt"
    if (Test-Path $latestPath) { $RunDir = (Get-Content $latestPath -Raw).Trim() }
}
Write-Host "---OIA FAST RUN---"
Write-Host $RunDir
Write-Host "---TASK---"
Get-ScheduledTask -TaskName 'SNNA_oia_fast_current' | Format-List TaskName,State
Get-ScheduledTaskInfo -TaskName 'SNNA_oia_fast_current' | Format-List LastRunTime,LastTaskResult
Write-Host "---LAUNCHER---"
if (Test-Path "$RunDir\launcher_meta.json") { Get-Content "$RunDir\launcher_meta.json" -Raw }
Write-Host "---ACTIVE WSL/PYTHON---"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'SNNA_repro|main_dino.py|multi_label_train.py|oia_fast' } |
    Select-Object ProcessId,Name,CommandLine | Format-List
Write-Host "---VRAM LAST---"
if (Test-Path "$RunDir\logs\vram_monitor.csv") { Get-Content "$RunDir\logs\vram_monitor.csv" -Tail 8 }
Write-Host "---DINO FAST TAIL---"
if (Test-Path "$RunDir\logs\dino_fast.log") { Get-Content "$RunDir\logs\dino_fast.log" -Tail 80 }
Write-Host "---BUILD DATASET TAIL---"
if (Test-Path "$RunDir\logs\build_oia_train_dino_dataset.log") { Get-Content "$RunDir\logs\build_oia_train_dino_dataset.log" -Tail 80 }
Write-Host "---CLASSIFIER FAST TAIL---"
if (Test-Path "$RunDir\logs\classifier_fast.log") { Get-Content "$RunDir\logs\classifier_fast.log" -Tail 80 }
Write-Host "---EXIT CODE---"
if (Test-Path "$RunDir\task_exit_code.txt") { Get-Content "$RunDir\task_exit_code.txt" -Raw }
