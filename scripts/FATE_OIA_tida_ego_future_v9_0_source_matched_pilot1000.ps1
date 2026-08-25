$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"
$output = "F:\FATE_Drive_runs\tida_ego_future_v9_0_source_matched_pilot1000"
New-Item -ItemType Directory -Force -Path $output | Out-Null
& "E:\Anaconda\envs\sbw39\python.exe" -u -m fate_oia.engine.train_tida_oia `
  --config "configs\fate_oia_train_tida_object_intent_v8_4_10k.yaml" `
  --clip-manifest "artifacts\tida_10k_v8\tida_source_matched_balanced_pilot1000_manifest.jsonl" `
  --image-checkpoint "F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth" `
  --checkpoint "F:\FATE_Drive_runs\tida_track_aligned_v8_9_source_matched_pilot1000\checkpoint_latest.pth" `
  --checkpoint-view online `
  --object-track-store "F:\FATE_Drive_runs\tida_object_tracks_source_matched1000_calib835_test885.pt" `
  --frame-store-root "F:\FATE_Drive_runs\tida_raw_frames_1000_calib324_test885" `
  --output-dir $output --epochs 4 --batch-size 6 --gradient-accumulation-steps 4 `
  --context-chunk-size 2 --num-workers 4 --max-samples 1000 `
  --max-calib-samples 835 --max-test-samples 885 --max-optimizer-updates 168 `
  --schedule-total-updates 168 --train-owners "object_intent_action,object_intent_reason" `
  --eval-every-epochs 4 --run-kind smoke --skip-ema-eval --skip-expanded-eval --device cuda
$code = $LASTEXITCODE
Set-Content -LiteralPath "$output\process_exit_code.txt" -Value $code -Encoding ascii
exit $code
