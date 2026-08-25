$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"

& "E:\Anaconda\envs\sbw39\python.exe" -u -m fate_oia.engine.extract_tida_object_tracks `
  --manifest "artifacts\tida_10k_v8\tida_10k_primary_manifest.jsonl" `
  --output "F:\FATE_Drive_runs\tida_object_tracks_1000_calib512_test512.pt" `
  --repository "E:\sbw\deps\co-tracker" `
  --train-limit 1000 `
  --eval-limit 512 `
  --partitions "train_core,train_calib,test" `
  --save-every 1 `
  --max-new-samples 20 `
  --num-workers 0 `
  --device cuda
