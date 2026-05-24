$ErrorActionPreference = 'SilentlyContinue'
$repo = 'E:\sbw\SNNA_repro\SNNA'
$runs = Get-ChildItem "$repo\repro_runs" -Directory -Filter 'smoke_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 3
foreach ($run in $runs) {
    Write-Host "---SMOKE RUN---"
    Write-Host $run.FullName
    $vram = Join-Path $run.FullName 'logs\vram_monitor.csv'
    if (Test-Path $vram) {
        Write-Host "---VRAM TAIL---"
        Get-Content $vram -Tail 10
        $max = 0
        Get-Content $vram | Select-Object -Skip 1 | ForEach-Object {
            $cols = $_ -split ','
            if ($cols.Count -ge 2) {
                $used = 0
                if ([int]::TryParse($cols[1].Trim(), [ref]$used)) {
                    if ($used -gt $max) { $max = $used }
                }
            }
        }
        Write-Host "max_vram_mib=$max"
    }
    $dino = Join-Path $run.FullName 'logs\dino_smoke.log'
    if (Test-Path $dino) {
        Write-Host "---DINO BATCH LINES---"
        Select-String -Path $dino -Pattern 'Epoch: \\[|Starting DINO|Data loaded|max mem' | Select-Object -Last 12
    }
    $cls = Join-Path $run.FullName 'logs\classifier_smoke.log'
    if (Test-Path $cls) {
        Write-Host "---CLASSIFIER BATCH LINES---"
        Select-String -Path $cls -Pattern 'Epoch: \\[|Data loaded|Accuracy|Averaged stats' | Select-Object -Last 12
    }
}
