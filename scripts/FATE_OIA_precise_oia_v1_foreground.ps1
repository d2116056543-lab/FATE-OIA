param(
  [ValidateSet("preflight", "pilot", "full")][string]$Mode = "preflight",
  [int]$Epochs = 12,
  [int]$BatchSize = 8,
  [int]$GradAccum = 4,
  [int]$NumWorkers = 8,
  [string]$Device = "cuda",
  [switch]$AllowFullWithEmbeddedCurriculum
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$argsList = @("-u", "-m", "fate_oia.engine.supervise_precise_oia_foreground", "--mode", $Mode, "--config", "configs\fate_oia_train_360x640_precise_oia_v1.yaml", "--output_dir", ".background_runs\precise_oia_v1_360x640_testprimary", "--epochs", "$Epochs", "--batch_size", "$BatchSize", "--gradient_accumulation_steps", "$GradAccum", "--num_workers", "$NumWorkers", "--device", "$Device")
if ($AllowFullWithEmbeddedCurriculum) {
  $argsList += "--allow_full_with_embedded_curriculum"
}
& E:\Anaconda\envs\sbw39\python.exe @argsList
exit $LASTEXITCODE
