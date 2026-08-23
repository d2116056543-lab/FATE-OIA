param(
    [string]$OutputDir = "F:\FATE_Drive_runs\tida_trajectory_v5_full_head_probe",
    [int]$Epochs = 1,
    [int]$BatchSize = 4,
    [int]$GradAccum = 8,
    [int]$NumWorkers = 4,
    [int]$ContextChunkSize = 2,
    [int]$MaxOptimizerUpdates = 77
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "E:\Anaconda\envs\sbw39\python.exe"
$log = Join-Path $OutputDir "head_probe.log"
$exitFile = Join-Path $OutputDir "process_exit_code.txt"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Set-Location $repo

# Windows PowerShell 5.1 wraps native stderr as NativeCommandError. DINO uses
# stderr for informational messages, so keep strict setup handling while the
# native training process is allowed to stream both channels into the log.
$ErrorActionPreference = "Continue"
& $python -u -m fate_oia.engine.train_tida_oia `
    --config configs\fate_oia_train_tida_oia_v5_trajectory.yaml `
    --clip-manifest E:\sbw\FATE_Drive\fate_oia_tida_oia_v1_worktree\artifacts\tida_clip_manifest.jsonl `
    --image-checkpoint F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth `
    --checkpoint F:\FATE_Drive_runs\tida_ctu_reason_ema_v5\checkpoint_best_test_joint.pth `
    --checkpoint-view ema `
    --output-dir $OutputDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --gradient-accumulation-steps $GradAccum `
    --context-chunk-size $ContextChunkSize `
    --num-workers $NumWorkers `
    --device cuda `
    --run-kind smoke `
    --max-optimizer-updates $MaxOptimizerUpdates `
    --schedule-total-updates $MaxOptimizerUpdates `
    --train-owners traffic_trajectory 2>&1 | Tee-Object -FilePath $log

$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Set-Content -LiteralPath $exitFile -Value $code -Encoding ascii
exit $code
