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
$pids = New-Object System.Collections.Generic.HashSet[int]
$pidText = (Get-Content -LiteralPath $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
if ($pidText) { [void]$pids.Add([int]$pidText) }
$escapedOut = [regex]::Escape((Resolve-Path -LiteralPath $OutputDir).Path)
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match $escapedOut -or $_.CommandLine -match 'fate_oia.engine.train_fate_oia' } |
  ForEach-Object { [void]$pids.Add([int]$_.ProcessId) }
if ($pids.Count -eq 0) {
  Write-Host "No FATE-OIA training processes found for OutputDir=$OutputDir"
  return
}
foreach ($pid in $pids) {
  $p = Get-Process -Id $pid -ErrorAction SilentlyContinue
  if ($p) {
    Stop-Process -Id $pid -Force
    Write-Host "Stopped FATE-OIA training PID=$pid"
  } else {
    Write-Host "Process PID=$pid is not running."
  }
}
