$ErrorActionPreference = "Stop"

$repo = "E:\sbw\FATE_Drive\fate_oia_tida_site_v20_worktree"
$outputDir = "E:\sbw\FATE_Drive\tida_v46_action_token_reason_smoke4"
$python = "E:\Anaconda\envs\sbw39\python.exe"

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Set-Location $repo
& $python -u -m fate_oia.engine.train_tida_oia `
    --config configs\fate_oia_train_tida_action_conditioned_reason_v43.yaml `
    --clip-manifest G:\FATE_Drive_runs\tida_full_manifest_exact4572_v19\tida_full_primary_manifest.jsonl `
    --image-checkpoint E:\sbw\FATE_Drive\fate_oia_acpr_pact_oia_v1_probe_worktree\.background_runs\pact_oia_v1_probe_control\checkpoint_epoch_000.pth `
    --output-dir $outputDir `
    --epochs 1 `
    --batch-size 1 `
    --gradient-accumulation-steps 2 `
    --context-chunk-size 3 `
    --num-workers 0 `
    --device cuda `
    --run-kind smoke `
    --max-samples 4 `
    --max-calib-samples 32 `
    --max-test-samples 4 `
    --max-audit-samples 4 `
    --train-owners action_local_query,reason_local_query `
    --skip-ema-eval `
    --skip-expanded-eval
exit $LASTEXITCODE
