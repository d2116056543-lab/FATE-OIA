$commandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree\scripts\Run_TIDA_v10_10_scale96_eval.ps1"
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $commandLine }
$result | ConvertTo-Json -Compress
