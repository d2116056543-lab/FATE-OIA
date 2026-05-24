param(
    [int]$DinoBatch = 8,
    [int]$ClassifierBatch = 4,
    [int]$DinoEpochs = 50,
    [int]$ClassifierEpochs = 100,
    [string]$DinoSplits = 'train,val,test',
    [string]$DinoResumeCheckpoint = ''
)

$ErrorActionPreference = 'Stop'
$repoWin = 'E:\sbw\SNNA_repro\SNNA'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runWin = Join-Path $repoWin "repro_runs\oia_fast_$stamp"
$logWin = Join-Path $runWin 'logs'
New-Item -ItemType Directory -Force -Path $logWin | Out-Null
Set-Content -LiteralPath (Join-Path $repoWin 'repro_runs\LATEST_SNNA_OIA_FAST.txt') -Value $runWin -Encoding ASCII

$repoWsl = '/mnt/e/sbw/SNNA_repro/SNNA'
$runWsl = "/mnt/e/sbw/SNNA_repro/SNNA/repro_runs/oia_fast_$stamp"
$launchWin = Join-Path $runWin 'launch_oia_fast.sh'
$launchWsl = "$runWsl/launch_oia_fast.sh"
$dinoDatasetTag = ($DinoSplits -replace '[^A-Za-z0-9]+', '_').Trim('_').ToLowerInvariant()
$launchText = @"
#!/usr/bin/env bash
set -euo pipefail
cd '$repoWsl'
REPO='$repoWsl' RUN_ROOT='$runWsl' DINO_BATCH='$DinoBatch' CLASSIFIER_BATCH='$ClassifierBatch' DINO_EPOCHS='$DinoEpochs' CLASSIFIER_EPOCHS='$ClassifierEpochs' DINO_SPLITS='$DinoSplits' DINO_DATASET_TAG='$dinoDatasetTag' DINO_RESUME_CHECKPOINT='$DinoResumeCheckpoint' bash scripts_repro/run_snna_oia_fast_pipeline.sh
"@
$launchText = $launchText -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($launchWin, $launchText, [System.Text.Encoding]::ASCII)

$meta = [ordered]@{
    run_win = $runWin
    run_wsl = $runWsl
    dino_batch = $DinoBatch
    classifier_batch = $ClassifierBatch
    dino_epochs = $DinoEpochs
    classifier_epochs = $ClassifierEpochs
    dino_splits = $DinoSplits
    dino_resume_checkpoint = $DinoResumeCheckpoint
    launch_script = $launchWin
    started_at = (Get-Date).ToString('s')
    task_mode = 'windows_scheduled_task_blocking_wsl'
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runWin 'launcher_meta.json') -Encoding UTF8

try {
    wsl.exe -d ADAPT-Ubuntu -- bash $launchWsl
    $code = $LASTEXITCODE
} catch {
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $logWin 'task_exception.log') -Encoding UTF8
    $code = 999
}

Set-Content -LiteralPath (Join-Path $runWin 'task_exit_code.txt') -Value $code -Encoding ASCII
exit $code
