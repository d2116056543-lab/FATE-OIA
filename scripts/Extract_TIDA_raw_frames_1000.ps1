$ErrorActionPreference = "Stop"

Set-Location "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"

$outputRoot = "F:\FATE_Drive_runs\tida_raw_frames_1000_calib324_test885"
$logRoot = "F:\FATE_Drive_runs\tida_raw_frame_extract_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

& "E:\Anaconda\envs\sbw39\python.exe" -u -m fate_oia.engine.extract_tida_raw_frames `
  --manifest "artifacts\tida_10k_v8\tida_10k_primary_manifest.jsonl" `
  --track-store "F:\FATE_Drive_runs\tida_object_tracks_1000_calib324_test885.pt" `
  --output-root $outputRoot `
  --workers 4 `
  --jpeg-quality 92 `
  2>&1 | Tee-Object -FilePath "$logRoot\extract.log" -Append

$exitCode = $LASTEXITCODE
Set-Content -LiteralPath "$logRoot\process_exit_code.txt" -Value $exitCode -Encoding ascii
exit $exitCode
