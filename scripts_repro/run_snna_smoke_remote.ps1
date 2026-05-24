param(
    [int]$DinoBatch = 16,
    [int]$ClassifierBatch = 4
)

$ErrorActionPreference = 'Stop'
$repoWsl = '/mnt/e/sbw/SNNA_repro/SNNA'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runWsl = "/mnt/e/sbw/SNNA_repro/SNNA/repro_runs/smoke_$stamp"
$cmd = "cd '$repoWsl' && REPO='$repoWsl' RUN_ROOT='$runWsl' DINO_BATCH='$DinoBatch' CLASSIFIER_BATCH='$ClassifierBatch' bash scripts_repro/run_snna_smoke_pipeline.sh"
wsl.exe -d ADAPT-Ubuntu -- bash -lc $cmd
if ($LASTEXITCODE -ne 0) {
    throw "SNNA smoke failed with exit code $LASTEXITCODE"
}
