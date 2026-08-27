$ErrorActionPreference = "Stop"
$worktree = "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"
$output = "G:\FATE_Drive_runs\tida_v10_6_proper_policy_eval1"
if (Test-Path -LiteralPath $output) { throw "Output already exists: $output" }
New-Item -ItemType Directory -Force -Path $output | Out-Null

$worker = Join-Path $worktree "scripts\Run_TIDA_v10_6_policy_eval.ps1"
$workerArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$worker`""
$process = Start-Process -FilePath "powershell.exe" `
  -ArgumentList $workerArguments -WorkingDirectory $worktree -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput "$output\launcher.out.log" `
  -RedirectStandardError "$output\launcher.err.log"
Set-Content -LiteralPath "$output\process_id.txt" -Value $process.Id -Encoding ascii
Write-Output "PID=$($process.Id) OUTPUT=$output"
