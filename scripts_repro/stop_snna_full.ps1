param(
    [string]$TaskName = 'SNNA_full_repro_current'
)
$ErrorActionPreference = 'SilentlyContinue'
Write-Host "---STOP TASK---"
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Format-List TaskName,State
Write-Host "---KILL WSL SNNA PROCESSES---"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "pkill -f 'main_dino.py|multi_label_train.py|run_snna_full_pipeline.sh|launch_full.sh' || true"
Start-Sleep -Seconds 3
Write-Host "---WINDOWS PROCESSES---"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'SNNA_repro|main_dino|multi_label_train|run_snna_full' } |
    Select-Object ProcessId,Name,CommandLine |
    Format-List
Write-Host "---GPU---"
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv
