param(
    [ValidateSet('smoke', 'full')]
    [string]$Mode = 'full',
    [string]$Config = 'configs\fate_oia_train_360x640_acpr_rael_oia_v1.yaml',
    [string]$OutputDir = '.background_runs\rael_oia_v1_full',
    [string]$Device = 'cuda',
    [int]$BatchSize = 6,
    [int]$GradAccum = 5,
    [int]$NumWorkers = 8,
    [int]$MaxTrainSamples = 0,
    [int]$MaxTestSamples = 0,
    [int]$MaxOptimizerUpdates = 0,
    [string]$FullGate = '.review\RAEL_OIA_V1_FULL_TRAIN_READY.json',
    [string]$RuntimeProfile = '.review\rael_oia_v1\runtime'
)

$ErrorActionPreference = 'Stop'
$Python = 'E:\Anaconda\envs\sbw39\python.exe'
if (-not (Test-Path $Python)) { throw "Missing Python: $Python" }
if ((git branch --show-current).Trim() -ne 'acpr_rael_oia_v1_direct_image') { throw 'Wrong RAEL branch.' }
if ($Mode -eq 'full') { $OutputDir = '.background_runs\rael_oia_v1_full' }
if ($Mode -eq 'smoke') { $OutputDir = '.background_runs\rael_oia_v1_smoke' }
$arguments = @('-u', '-m', 'fate_oia.engine.supervise_acpr_rael_oia_foreground', '--mode', $Mode, '--config', $Config, '--output_dir', $OutputDir, '--device', $Device, '--batch_size', $BatchSize, '--gradient_accumulation_steps', $GradAccum, '--num_workers', $NumWorkers)
if ($MaxTrainSamples -gt 0) { $arguments += @('--max_train_samples', $MaxTrainSamples) }
if ($MaxTestSamples -gt 0) { $arguments += @('--max_test_samples', $MaxTestSamples) }
if ($MaxOptimizerUpdates -gt 0) { $arguments += @('--max_optimizer_updates', $MaxOptimizerUpdates) }
if ($Mode -eq 'full') {
    $arguments += @('--full_gate', $FullGate, '--runtime_profile', $RuntimeProfile)
}
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "RAEL foreground supervisor failed with exit code $LASTEXITCODE" }
