@echo off
start "" /b powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree\scripts\Prepare_TIDA_source_complete_missing272.ps1" ^> "G:\FATE_Drive_runs\tida_source_complete_launcher.log" 2^>^&1
exit /b 0
