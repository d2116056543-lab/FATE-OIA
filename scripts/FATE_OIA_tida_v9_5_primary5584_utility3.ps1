$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"

$output = "G:\FATE_Drive_runs\tida_v9_5_primary5584_utility3"
if (Test-Path $output) {
  throw "Output already exists: $output"
}
New-Item -ItemType Directory -Force -Path $output | Out-Null

& "E:\Anaconda\envs\sbw39\python.exe" -u -m fate_oia.engine.train_tida_oia `
  --config "configs\fate_oia_train_tida_object_intent_v8_4_10k.yaml" `
  --clip-manifest "artifacts\tida_10k_v8\tida_10k_primary_manifest.jsonl" `
  --image-checkpoint "F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth" `
  --checkpoint "G:\FATE_Drive_runs\tida_v9_4_primary5584_route\checkpoint_latest.pth" `
  --checkpoint-view online `
  --object-track-store "G:\FATE_Drive_runs\tida_object_tracks_primary5584_calib779_test885.pt" `
  --frame-store-root "G:\FATE_Drive_runs\tida_raw_frames_primary5584_calib779_test885" `
  --output-dir $output --epochs 3 --batch-size 4 --gradient-accumulation-steps 6 `
  --context-chunk-size 2 --num-workers 4 --max-samples 5584 `
  --max-calib-samples 779 --max-test-samples 885 --max-optimizer-updates 699 `
  --schedule-total-updates 699 `
  --train-owners "object_intent_action_utility,object_intent_reason_utility" `
  --eval-every-epochs 1 --run-kind full --skip-ema-eval --skip-expanded-eval --device cuda

$code = $LASTEXITCODE
Set-Content -LiteralPath "$output\process_exit_code.txt" -Value $code -Encoding ascii
exit $code
