param(
    [string]$RunDir = ''
)

$ErrorActionPreference = 'SilentlyContinue'
$repo = 'E:\sbw\SNNA_repro\SNNA'
Set-Location $repo
if (-not $RunDir) {
    $latest = Get-ChildItem "$repo\repro_runs" -Directory -Filter 'full_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { $RunDir = $latest.FullName }
}
Write-Host "---SNNA RUN---"
Write-Host $RunDir
Write-Host "---LAUNCHER---"
if (Test-Path "$RunDir\launcher_meta.json") { Get-Content "$RunDir\launcher_meta.json" -Raw }
Write-Host "---ACTIVE WSL/PYTHON---"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'run_snna_full_pipeline|main_dino.py|multi_label_train.py|SNNA_repro' } |
    Select-Object ProcessId,Name,CommandLine | Format-List
Write-Host "---VRAM LAST---"
if (Test-Path "$RunDir\logs\vram_monitor.csv") { Get-Content "$RunDir\logs\vram_monitor.csv" -Tail 8 }
Write-Host "---DINO FULL TAIL---"
if (Test-Path "$RunDir\logs\dino_full.log") { Get-Content "$RunDir\logs\dino_full.log" -Tail 60 }
Write-Host "---CLASSIFIER FULL TAIL---"
if (Test-Path "$RunDir\logs\classifier_full.log") { Get-Content "$RunDir\logs\classifier_full.log" -Tail 60 }
Write-Host "---STDOUT TAIL---"
if (Test-Path "$RunDir\logs\full_pipeline_stdout.log") { Get-Content "$RunDir\logs\full_pipeline_stdout.log" -Tail 40 }
Write-Host "---STDERR TAIL---"
if (Test-Path "$RunDir\logs\full_pipeline_stderr.log") { Get-Content "$RunDir\logs\full_pipeline_stderr.log" -Tail 40 }
Write-Host "---NOHUP TAIL---"
if (Test-Path "$RunDir\logs\nohup.out") { Get-Content "$RunDir\logs\nohup.out" -Tail 80 }
Write-Host "---WSL PID---"
if (Test-Path "$RunDir\wsl_pid.txt") { Get-Content "$RunDir\wsl_pid.txt" -Raw }
