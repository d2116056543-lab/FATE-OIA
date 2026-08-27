$commandLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree\scripts\FATE_OIA_tida_v10_11_full_boundary_reader1.ps1"
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $commandLine }
$result | ConvertTo-Json -Compress
