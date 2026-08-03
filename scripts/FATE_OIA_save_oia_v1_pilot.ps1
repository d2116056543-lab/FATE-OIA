param(
  [string]$Device = "cuda",
  [string]$OutputDir = ".background_runs\save_oia_v1_pilot",
  [string]$PreflightDir = ".background_runs\save_oia_v1_preflight",
  [int]$NumWorkers = 4
)

$ErrorActionPreference = "Stop"
$Python = "E:\Anaconda\envs\sbw39\python.exe"
$Config = "configs\fate_oia_train_360x640_save_oia_v1.yaml"

& $Python -u -m fate_oia.engine.audit_save_oia `
  --config $Config --output-dir $PreflightDir --device $Device `
  --write-review-pass
if ($LASTEXITCODE -ne 0) { throw "SAVE implementation audit failed" }

& $Python -u -m fate_oia.engine.profile_save_oia `
  --config $Config --output-dir $PreflightDir --device $Device
if ($LASTEXITCODE -ne 0) { throw "SAVE real-DINO runtime profile failed" }

& $Python -u -m fate_oia.engine.train_save_oia `
  --config $Config --output-dir $OutputDir --run-kind pilot `
  --epochs 4 --max-train-samples 4096 --max-audit-samples 1024 `
  --max-calib-samples 512 --max-test-samples 512 --seed 20260803 `
  --num-workers $NumWorkers --device $Device
if ($LASTEXITCODE -ne 0) { throw "SAVE four-epoch pilot failed" }

& $Python -u -m fate_oia.engine.evaluate_save_oia_pilot `
  --raw-evidence "$OutputDir\save_pilot_raw_evidence_input.json" `
  --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "SAVE pilot gates A-G failed" }
