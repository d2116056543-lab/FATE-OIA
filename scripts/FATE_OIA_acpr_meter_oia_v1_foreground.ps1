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
$Ready = Join-Path $Root '.review\METER_OIA_V1_PRE_PILOT_READY.json'
if (-not (Test-Path $Ready)) { throw "PRE_PILOT readiness artifact is required before $Mode" }
$args = @('-u','-m','fate_oia.engine.train_acpr_meter_oia','--config',$Config,'--output_dir',$OutputDir,'--device',$Device,'--require_ready','--worktree_root',$Root)
if ($Mode -eq 'pilot') {
  $args += @('--epochs','3','--max_train_samples','2048','--max_audit_samples','128','--max_calib_samples','128','--max_test_samples','128')
} else {
  $args += @('--epochs','12')
}
if ($UseMockDino) { $args += '--use_mock_dino' }
& $Python @args
if ($LASTEXITCODE -ne 0) { throw "METER foreground training exited with code $LASTEXITCODE" }
