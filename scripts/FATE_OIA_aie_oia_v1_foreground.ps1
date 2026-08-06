param([int]$Epochs=20, [string]$Device="cuda", [string]$OutputDir="")
$ErrorActionPreference = "Stop"
$head = (git rev-parse --short HEAD).Trim()
if (-not $OutputDir) { $OutputDir = "E:\FATE_OIA_aie_oia_v1_full_$head" }
& E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_aie_oia_foreground `
  --config configs\fate_oia_train_360x640_aie_oia_v1.yaml `
  --output-dir $OutputDir --run-kind full --epochs $Epochs --device $Device
exit $LASTEXITCODE

