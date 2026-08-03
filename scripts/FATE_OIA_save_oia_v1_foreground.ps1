param(
  [string]$Device = "cuda",
  [string]$OutputDir = "E:\FATE_OIA_save_oia_v1_full",
  [string]$PreflightDir = ".background_runs\save_oia_v1_preflight",
  [string]$PilotDir = ".background_runs\save_oia_v1_pilot",
  [int]$NumWorkers = 4,
  [switch]$AllowNumericCandidate
)

$ErrorActionPreference = "Stop"
$Python = "E:\Anaconda\envs\sbw39\python.exe"
& $Python -u -m fate_oia.engine.supervise_save_oia_foreground `
  --config configs\fate_oia_train_360x640_save_oia_v1.yaml `
  --output-dir $OutputDir --device $Device --num-workers $NumWorkers `
  --review "$PreflightDir\SAVE_IMPLEMENTATION_REVIEW.json" `
  --profile "$PreflightDir\SAVE_RUNTIME_PROFILE.json" `
  --pilot "$PilotDir\SAVE_PILOT_PASS.json" `
  --ready "$PilotDir\SAVE_FULL_TRAIN_READY.json" `
  $(if ($AllowNumericCandidate) { "--allow-numeric-candidate" })
if ($LASTEXITCODE -ne 0) { throw "SAVE foreground full training exited with code $LASTEXITCODE" }
