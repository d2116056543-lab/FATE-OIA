$pattern = "*tida_v10_11_full_boundary_reader1*"
$processes = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pattern -and $_.Name -eq "python.exe" }
Write-Output ("alive=" + [bool]$processes + " pids=" + (($processes.ProcessId -join ",")))
foreach ($process in $processes) {
    $live = Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue
    if ($live) { Write-Output ("cpu_sec=" + [math]::Round($live.CPU, 1) + " working_set_gb=" + [math]::Round($live.WorkingSet64 / 1GB, 2)) }
}
$log = "G:\FATE_Drive_runs\tida_v10_11_full_boundary_reader1\full_reader.log"
if (Test-Path $log) {
    $eventLines = Select-String -Path $log -Pattern '^\{"event"' | Select-Object -Last 3
    foreach ($match in $eventLines) {
        try {
            $event = $match.Line | ConvertFrom-Json
            if ($event.event -eq "tida_batch") {
                Write-Output ("batch epoch=" + $event.epoch + " update=" + $event.optimizer_update + "/" + $event.total_updates + " loss=" + [math]::Round($event.loss_total, 5) + " helpful=" + [math]::Round($event.object_intent_action_candidate_helpful_rate, 4) + " cand_rms=" + [math]::Round($event.object_intent_action_candidate_rms, 6) + " sps=" + [math]::Round($event.samples_per_second, 3))
            } elseif ($event.event -eq "tida_epoch") {
                Write-Output ("epoch result action_mf1=" + $event.action_mf1 + " action_map=" + $event.action_map + " exp_mf1=" + $event.exp_mf1 + " exp_map=" + $event.exp_map)
            }
        } catch { }
    }
    $anomalies = Select-String -Path $log -Pattern 'NaN|nan|OOM|out of memory|Traceback' | Select-Object -Last 3
    $anomalies | ForEach-Object { $_.Line }
    $item = Get-Item $log
    Write-Output ("log_bytes=" + $item.Length + " log_age_sec=" + [math]::Round(((Get-Date) - $item.LastWriteTime).TotalSeconds, 1))
}
$exitFile = "G:\FATE_Drive_runs\tida_v10_11_full_boundary_reader1\process_exit_code.txt"
if (Test-Path $exitFile) { Write-Output ("exit_code=" + (Get-Content -Raw $exitFile).Trim()) }
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
