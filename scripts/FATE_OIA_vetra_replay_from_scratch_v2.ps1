param(
    [string]$OutputDir = "F:\FATE_Drive_runs\vetra_replay_from_scratch_v2_full",
    [int]$BatchSize = 6,
    [int]$GradAccum = 5,
    [int]$NumWorkers = 8,
    [string]$Device = "cuda",
    [switch]$Resume,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "E:\Anaconda\envs\sbw39\python.exe"
$arguments = @(
    "-u", "-m", "fate_oia.engine.supervise_vetra_replay_from_scratch",
    "--config", "configs\fate_oia_train_360x640_vetra_replay_from_scratch_v2.yaml",
    "--output-dir", $OutputDir,
    "--python", $python,
    "--batch-size", $BatchSize,
    "--gradient-accumulation-steps", $GradAccum,
    "--num-workers", $NumWorkers,
    "--device", $Device
)
if ($Resume) { $arguments += "--resume" }
if ($Smoke) { $arguments += "--smoke" }

Push-Location $repo
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "VETRA replay supervisor exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
