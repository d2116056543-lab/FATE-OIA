$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "E:\Anaconda\envs\sbw39\python.exe"
$output = "F:\FATE_Drive_runs\tida_trajectory_v6_3_utility_only"
$checkpoint = "F:\FATE_Drive_runs\tida_trajectory_v6_0_relational_continue\checkpoint_epoch_001.pth"
$log = Join-Path $output "utility_only.log"
$exitFile = Join-Path $output "process_exit_code.txt"
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Location $repo

$ErrorActionPreference = "Continue"
& $python -u -m fate_oia.engine.train_tida_oia `
    --config configs\fate_oia_train_tida_oia_v5_trajectory.yaml `
    --clip-manifest E:\sbw\FATE_Drive\fate_oia_tida_oia_v1_worktree\artifacts\tida_clip_manifest.jsonl `
    --image-checkpoint F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth `
    --checkpoint $checkpoint `
    --checkpoint-view online `
    --output-dir $output `
    --epochs 2 `
    --batch-size 6 `
    --gradient-accumulation-steps 5 `
    --context-chunk-size 2 `
    --num-workers 6 `
    --device cuda `
    --run-kind smoke `
    --max-optimizer-updates 154 `
    --schedule-total-updates 154 `
    --train-owners traffic_trajectory_utility 2>&1 | Tee-Object -FilePath $log

$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Set-Content -LiteralPath $exitFile -Value $code -Encoding ascii
exit $code
