param(
  [string]$OutputDir = "F:\FATE_Drive_runs\vetra_staged_from_scratch_v1",
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [int]$NumWorkers = 8,
  [string]$Device = "cuda",
  [switch]$Resume,
  [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$Python = "E:\Anaconda\envs\sbw39\python.exe"
$Arguments = @(
  "-u", "-m", "fate_oia.engine.supervise_vetra_staged_from_scratch",
  "--config", "configs\fate_oia_train_360x640_vetra_staged_from_scratch_v1.yaml",
  "--output-dir", $OutputDir,
  "--python", $Python,
  "--batch-size", $BatchSize,
  "--gradient-accumulation-steps", $GradAccum,
  "--num-workers", $NumWorkers,
  "--device", $Device
)
if ($Resume) { $Arguments += "--resume" }
if ($Smoke) { $Arguments += "--smoke" }
& $Python @Arguments
exit $LASTEXITCODE
