param(
  [string]$Run = "status"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here
Set-Location $Repo

$matrix = @(
  [pscustomobject]@{Run="A"; Script="FATE_OIA_runA_e13_cosine_lr1e4.ps1"; Purpose="low LR + cosine baseline; already run and stopped if plateaued"},
  [pscustomobject]@{Run="B"; Script="FATE_OIA_runB_e13_const_lr1e4.ps1"; Purpose="constant lr=1e-4 control for scheduler effect"},
  [pscustomobject]@{Run="C"; Script="FATE_OIA_runC_e13_cosine_labelcorr.ps1"; Purpose="LabelCorrelationBlock only; active if already launched"},
  [pscustomobject]@{Run="D"; Script="FATE_OIA_runD_e13_cosine_labelcorr_uncertainty.ps1"; Purpose="LabelCorrelationBlock + uncertainty task balancing"}
)

if ($Run -eq "status") {
  Write-Host "Controlled FATE-OIA experiment matrix"
  $matrix | Format-Table -AutoSize
  Write-Host ""
  Write-Host "Active train_fate_oia processes:"
  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'fate_oia\.engine\.train_fate_oia' } |
    Select-Object ProcessId,Name,CommandLine | Format-List
  Write-Host ""
  Write-Host "Run command examples:"
  Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\FATE_OIA_runB_e13_const_lr1e4.ps1"
  Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\FATE_OIA_runC_e13_cosine_labelcorr.ps1"
  Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\FATE_OIA_runD_e13_cosine_labelcorr_uncertainty.ps1"
  exit 0
}

$selected = $matrix | Where-Object { $_.Run -ieq $Run }
if (-not $selected) {
  throw "Unknown run '$Run'. Use A, B, C, D, or status."
}
& (Join-Path $Here $selected.Script)
