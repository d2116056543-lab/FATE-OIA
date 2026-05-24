param(
    [int]$DinoBatch = 8,
    [int]$ClassifierBatch = 4
)

$ErrorActionPreference = 'Stop'
$repoWin = 'E:\sbw\SNNA_repro\SNNA'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runWin = Join-Path $repoWin "repro_runs\full_$stamp"
$logWin = Join-Path $runWin 'logs'
New-Item -ItemType Directory -Force -Path $logWin | Out-Null

$repoWsl = '/mnt/e/sbw/SNNA_repro/SNNA'
$runWsl = "/mnt/e/sbw/SNNA_repro/SNNA/repro_runs/full_$stamp"
$launchWin = Join-Path $runWin 'launch_full.sh'
$launchWsl = "$runWsl/launch_full.sh"
$launchText = @"
#!/usr/bin/env bash
set -euo pipefail
cd '$repoWsl'
REPO='$repoWsl' RUN_ROOT='$runWsl' DINO_BATCH='$DinoBatch' CLASSIFIER_BATCH='$ClassifierBatch' bash scripts_repro/run_snna_full_pipeline.sh
"@
$launchText = $launchText -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($launchWin, $launchText, [System.Text.Encoding]::ASCII)

$cmd = "chmod +x '$launchWsl'; nohup bash '$launchWsl' > '$runWsl/logs/nohup.out' 2>&1 < /dev/null & echo `$! > '$runWsl/wsl_pid.txt'"
wsl.exe -d ADAPT-Ubuntu -- bash -lc $cmd
if ($LASTEXITCODE -ne 0) {
    throw "WSL nohup launch failed with exit code $LASTEXITCODE"
}

$meta = [ordered]@{
    run_win = $runWin
    run_wsl = $runWsl
    wsl_pid_file = Join-Path $runWin 'wsl_pid.txt'
    dino_batch = $DinoBatch
    classifier_batch = $ClassifierBatch
    launch_script = $launchWin
    stdout = Join-Path $logWin 'nohook'
    nohup_out = Join-Path $logWin 'nohup.out'
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runWin 'launcher_meta.json') -Encoding UTF8
$meta | ConvertTo-Json -Depth 4
