$ErrorActionPreference = "Stop"

$repo = "E:\sbw\FATE_Drive\fate_oia_tida_site_v20_worktree"
$outputDir = "E:\sbw\FATE_Drive\tida_v46_action_token_reason_pilot4096"
$python = "E:\Anaconda\envs\sbw39\python.exe"

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
"launcher_entered $(Get-Date -Format o)" | Set-Content "$outputDir\launcher_status.txt"
Set-Location $repo
$ErrorActionPreference = "Continue"
& $python -u -m fate_oia.engine.train_tida_oia `
    --config configs\fate_oia_train_tida_action_conditioned_reason_v43.yaml `
    --clip-manifest G:\FATE_Drive_runs\tida_full_manifest_exact4572_v19\tida_full_primary_manifest.jsonl `
    --image-checkpoint E:\sbw\FATE_Drive\fate_oia_acpr_pact_oia_v1_probe_worktree\.background_runs\pact_oia_v1_probe_control\checkpoint_epoch_000.pth `
    --output-dir $outputDir `
    --epochs 1 `
    --batch-size 6 `
    --gradient-accumulation-steps 5 `
    --context-chunk-size 3 `
    --num-workers 4 `
    --device cuda `
    --run-kind smoke `
    --max-samples 4096 `
    --max-calib-samples 1024 `
    --max-test-samples 4572 `
    --max-audit-samples 128 `
    --train-owners action_local_query,reason_local_query `
    --verified-baseline-artifact F:\FATE_Drive_runs\tida_v39_target_track_pilot4096\TIDA_IMAGE_BASELINE_COVERED_SUBSET.json `
    --skip-ema-eval `
    --skip-expanded-eval `
    *> "$outputDir\train.log"
$code = $LASTEXITCODE
$code | Set-Content "$outputDir\process_exit_code.txt"
"python_exited code=$code $(Get-Date -Format o)" | Add-Content "$outputDir\launcher_status.txt"
exit $code
