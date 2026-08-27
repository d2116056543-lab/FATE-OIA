$ErrorActionPreference = "Continue"
$repo = "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"
$python = "E:\Anaconda\envs\sbw39\python.exe"
$artifact = Join-Path $repo "artifacts\tida_10k_v10_source_complete"
$runRoot = "G:\FATE_Drive_runs"
$trackStore = Join-Path $runRoot "tida_object_tracks_source_complete_missing272.pt"
$frameRoot = Join-Path $runRoot "tida_raw_frames_source_complete"
$log = Join-Path $runRoot "tida_source_complete_precompute.log"

Set-Location $repo
$env:PYTHONPATH = "."
& $python -u -m fate_oia.engine.extract_tida_object_tracks `
  --manifest (Join-Path $artifact "missing_track_manifest.jsonl") `
  --output $trackStore `
  --repository "E:\sbw\deps\co-tracker" `
  --train-limit 10000 `
  --eval-limit 10000 `
  --partitions "train_core,train_calib,train_audit" `
  --num-workers 0 `
  --save-every 25 `
  --device cuda *>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -u -m fate_oia.engine.extract_tida_raw_frames `
  --manifest (Join-Path $artifact "missing_frame_manifest.jsonl") `
  --track-store $trackStore `
  --output-root $frameRoot `
  --workers 6 *>&1 | Tee-Object -FilePath $log -Append
exit $LASTEXITCODE
