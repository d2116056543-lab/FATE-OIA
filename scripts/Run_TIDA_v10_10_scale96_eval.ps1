$ErrorActionPreference = "Continue"
$worktree = "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"
$output = "G:\FATE_Drive_runs\tida_v10_10_scale96_eval1"
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Location $worktree
$env:PYTHONPATH = "."

& "E:\Anaconda\envs\sbw39\python.exe" -u -m fate_oia.engine.train_tida_oia `
  --config "configs\fate_oia_train_tida_object_intent_v8_4_10k.yaml" `
  --clip-manifest "artifacts\tida_10k_v10_source_complete\tida_source_complete_manifest.jsonl" `
  --image-checkpoint "F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth" `
  --checkpoint "G:\FATE_Drive_runs\tida_v10_9_boundary_reader_pilot1\checkpoint_best_test_action_mf1.pth" `
  --checkpoint-view online `
  --object-track-store "G:\FATE_Drive_runs\tida_object_tracks_source_complete8916.pt" `
  --frame-store-root "G:\FATE_Drive_runs\tida_raw_frames_primary5584_calib779_test885;F:\FATE_Drive_runs\tida_raw_frames_1000_calib324_test885;G:\FATE_Drive_runs\tida_raw_frames_source_complete" `
  --verified-baseline-artifact "G:\FATE_Drive_runs\tida_v10_source_complete_utility2\TIDA_IMAGE_BASELINE_COVERED_SUBSET.json" `
  --output-dir $output --epochs 1 --batch-size 6 --gradient-accumulation-steps 4 `
  --context-chunk-size 2 --num-workers 6 --max-samples 2048 `
  --max-calib-samples 838 --max-audit-samples 1128 --max-test-samples 885 `
  --evaluation-only --policy-use-train-core --run-kind smoke `
  --skip-ema-eval --skip-expanded-eval --device cuda `
  *> "$output\evaluation.log"
Set-Content -LiteralPath "$output\process_exit_code.txt" -Value $LASTEXITCODE -Encoding ascii
