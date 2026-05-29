param(
  [string]$Device = "cuda",
  [int]$MaxTestSamples = 0,
  [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
if ([string]::IsNullOrWhiteSpace($Output)) {
  $Output = Join-Path $Repo "runc_outputs\current_code_eval_repro.json"
}
Set-Location $Repo
& "E:\Anaconda\envs\sbw39\python.exe" "tools\eval_runc_current_code.py" `
  --device $Device `
  --max_test_samples $MaxTestSamples `
  --output $Output
exit $LASTEXITCODE
