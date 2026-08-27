$pattern = "*tida_v10_10_scale96_eval1*"
$processes = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pattern -and $_.Name -eq "python.exe" }
Write-Output ("alive=" + [bool]$processes + " pids=" + (($processes.ProcessId -join ",")))
foreach ($process in $processes) {
    $live = Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
    if ($live) {
        Write-Output ("cpu_sec=" + [math]::Round($live.CPU, 1) + " working_set_gb=" + [math]::Round($live.WorkingSet64 / 1GB, 2))
    }
}
$log = "G:\FATE_Drive_runs\tida_v10_10_scale96_eval1\evaluation.log"
if (Test-Path $log) {
    Get-Content -Tail 12 $log
    $item = Get-Item $log
    Write-Output ("log_bytes=" + $item.Length)
    Write-Output ("log_age_sec=" + [math]::Round(((Get-Date) - $item.LastWriteTime).TotalSeconds, 1))
}
$exitFile = "G:\FATE_Drive_runs\tida_v10_10_scale96_eval1\process_exit_code.txt"
if (Test-Path $exitFile) { Write-Output ("exit_code=" + (Get-Content -Raw $exitFile).Trim()) }
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
