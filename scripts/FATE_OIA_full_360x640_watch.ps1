param(
  [string]$OutputDir = $env:FATE_OIA_OUTPUT_DIR,
  [int]$Tail = 80,
  [switch]$Wait
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
if (-not $OutputDir) {
  $latest = Get-ChildItem -LiteralPath ".background_runs" -Directory -Filter "fate_oia_full_360x640_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { throw "No fate_oia_full_360x640_* run found." }
  $OutputDir = $latest.FullName
}
$log = Join-Path $OutputDir "train.log"
$err = Join-Path $OutputDir "train.stderr.log"
Write-Host "OutputDir=$OutputDir"
if (Test-Path (Join-Path $OutputDir "train.pid")) { Write-Host "PID=$(Get-Content (Join-Path $OutputDir 'train.pid'))" }
if (Test-Path $err) { Write-Host "--- stderr tail ---"; Get-Content -LiteralPath $err -Tail 40 }
Write-Host "--- train tail ---"
if ($Wait) { Get-Content -LiteralPath $log -Tail $Tail -Wait } else { Get-Content -LiteralPath $log -Tail $Tail }
