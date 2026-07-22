param(
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $WorkingDirectory
$args = @(
    "-u", "-m", "fate_oia.engine.train_acpr_mosaic_trust_icdor",
    "--config", "configs/fate_oia_train_360x640_acpr_mosaic_trust_v5_credo_map_pilot.yaml",
    "--output_dir", ".background_runs/mosaic_v5_full_unreviewed_b4_20260720",
    "--runtime_selection", ".review/mosaic_icdor_runtime_selection.json",
    "--epochs", "12", "--batch_size", "4", "--gradient_accumulation_steps", "8",
    "--num_workers", "4", "--device", "cuda", "--allow_unreviewed_full_train", "--seed", "20260720"
)
& $PythonExe @args
exit $LASTEXITCODE
