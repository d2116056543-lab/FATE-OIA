param(
  [string]$Config = "configs\fate_oia_train_360x640_vetra_oia_v1_probe.yaml",
  [Parameter(Mandatory=$true)][string]$SourceCheckpoint,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [int]$Epochs = 3,
  [switch]$Screening
)
$ErrorActionPreference = "Stop"
$args = @("-u", "-m", "fate_oia.engine.supervise_vetra_oia_probe", "--config", $Config,
  "--source-checkpoint", $SourceCheckpoint, "--output-dir", $OutputDir, "--epochs", "$Epochs", "--device", "cuda")
if ($Screening) { $args += "--screening" }
& "E:\Anaconda\envs\sbw39\python.exe" @args
if ($LASTEXITCODE -ne 0) { throw "VETRA supervisor failed with exit code $LASTEXITCODE" }
