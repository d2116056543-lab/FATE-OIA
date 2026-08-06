param([string]$Device="cuda")
$ErrorActionPreference = "Stop"
$head = (git rev-parse --short HEAD).Trim()
& E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_aie_oia_foreground `
  --config configs\fate_oia_train_360x640_aie_oia_v1.yaml `
  --output-dir ".background_runs\aie_oia_v1_pilot_$head" `
  --run-kind pilot --epochs 4 --device $Device `
  --max-train-samples 4096 --max-audit-samples 1024 `
  --max-calib-samples 512 --max-test-samples 512
exit $LASTEXITCODE
