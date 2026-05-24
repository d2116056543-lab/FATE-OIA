param(
    [int]$DinoBatch = 16,
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

$stdout = Join-Path $logWin 'full_pipeline_stdout.log'
$stderr = Join-Path $logWin 'full_pipeline_stderr.log'
$proc = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d','ADAPT-Ubuntu','--','bash',$launchWsl) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$meta = [ordered]@{
    run_win = $runWin
    run_wsl = $runWsl
    pid = $proc.Id
    dino_batch = $DinoBatch
    classifier_batch = $ClassifierBatch
    launch_script = $launchWin
    stdout = $stdout
    stderr = $stderr
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runWin 'launcher_meta.json') -Encoding UTF8
$meta | ConvertTo-Json -Depth 4
