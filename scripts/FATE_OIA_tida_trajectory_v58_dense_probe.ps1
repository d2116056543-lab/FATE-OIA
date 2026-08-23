$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "FATE_OIA_tida_trajectory_v5_head_probe.ps1") `
    -OutputDir "F:\FATE_Drive_runs\tida_trajectory_v5_8_dense_local_probe" `
    -Epochs 1 `
    -BatchSize 4 `
    -GradAccum 8 `
    -NumWorkers 4 `
    -ContextChunkSize 2 `
    -MaxOptimizerUpdates 77
exit $LASTEXITCODE
