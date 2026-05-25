param([string]$OutputDir = $env:FATE_OIA_OUTPUT_DIR)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
if (-not $OutputDir) {
  $latest = Get-ChildItem -LiteralPath ".background_runs" -Directory -Filter "fate_oia_full_360x640_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { throw "No fate_oia_full_360x640_* run found." }
  $OutputDir = $latest.FullName
}
$pidPath = Join-Path $OutputDir "train.pid"
if (-not (Test-Path $pidPath)) { throw "PID file not found: $pidPath" }
$pid = [int](Get-Content -LiteralPath $pidPath)
$p = Get-Process -Id $pid -ErrorAction SilentlyContinue
if ($p) {
  Stop-Process -Id $pid -Force
  Write-Host "Stopped FATE-OIA training PID=$pid"
} else {
  Write-Host "Process PID=$pid is not running."
}
