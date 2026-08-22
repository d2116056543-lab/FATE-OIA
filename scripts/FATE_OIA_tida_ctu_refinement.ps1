param(
    [string]$OutputDir = "F:\FATE_Drive_runs\tida_ctu_refinement_v2",
    [string]$Checkpoint = "F:\FATE_Drive_runs\tida_flow_credit_full_362d540\checkpoint_best_test_joint.pth",
    [int]$Epochs = 3,
    [int]$BatchSize = 4,
    [int]$GradAccum = 8,
    [int]$NumWorkers = 6,
    [string]$TrainOwners = "",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "E:\Anaconda\envs\sbw39\python.exe"
$manifest = "E:\sbw\FATE_Drive\fate_oia_tida_oia_v1_worktree\artifacts\tida_clip_manifest.jsonl"
$imageCheckpoint = "F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full_20260819_retry1\checkpoint_stage_b_continued.pth"
$microBatches = [Math]::Ceiling(2291.0 / $BatchSize)
$scheduleUpdates = [int]([Math]::Ceiling($microBatches / $GradAccum) * $Epochs)

Set-Location $root
$ownerArgs = @()
if ($TrainOwners) {
    $ownerArgs = @("--train-owners", $TrainOwners)
}

& $python -u -m fate_oia.engine.train_tida_oia `
    --config configs\fate_oia_train_tida_oia_v1_15f.yaml `
    --clip-manifest $manifest `
    --image-checkpoint $imageCheckpoint `
    --checkpoint $Checkpoint `
    --output-dir $OutputDir `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --gradient-accumulation-steps $GradAccum `
    --context-chunk-size 2 `
    --num-workers $NumWorkers `
    --schedule-total-updates $scheduleUpdates `
    --run-kind smoke `
    --device $Device `
    @ownerArgs

if ($LASTEXITCODE -ne 0) {
    throw "TIDA CTU refinement failed with exit code $LASTEXITCODE"
}
