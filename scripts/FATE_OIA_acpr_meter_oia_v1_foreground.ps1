param(
  [ValidateSet('pilot','full')][string]$Mode = 'pilot',
  [string]$Config = 'configs\fate_oia_train_360x640_acpr_meter_oia_v1.yaml',
  [string]$OutputDir = '',
  [string]$Device = 'cuda',
  [switch]$UseMockDino
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Python = 'E:\Anaconda\envs\sbw39\python.exe'
if (-not (Test-Path $Python)) { throw "Missing Python runtime: $Python" }
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = if ($Mode -eq 'pilot') { '.background_runs\meter_oia_v1_pilot' } else { '.background_runs\meter_oia_v1_full' }
}
if ($UseMockDino) { throw 'METER pilot/full supervisor requires real DINO' }
$ReadyName = if ($Mode -eq 'pilot') { 'METER_OIA_V1_PRE_PILOT_READY.json' } else { 'METER_OIA_V1_FULL_TRAIN_READY.json' }
$Ready = Join-Path $Root (Join-Path '.review' $ReadyName)
if (-not (Test-Path $Ready)) { throw "$ReadyName is required before $Mode" }
$args = @(
  '-u','-m','fate_oia.engine.supervise_acpr_meter_oia_foreground',
  '--mode',$Mode,
  '--config',$Config,
  '--output_dir',$OutputDir,
  '--device',$Device,
  '--pilot_train_samples','4096',
  '--pilot_audit_samples','1024',
  '--pilot_calib_samples','512',
  '--pilot_test_samples','512'
)
& $Python @args
if ($LASTEXITCODE -ne 0) { throw "METER foreground training exited with code $LASTEXITCODE" }
