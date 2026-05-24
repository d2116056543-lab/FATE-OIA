$repo = 'E:\sbw\SNNA_repro\SNNA'
Write-Host "---LATEST FILE---"
if (Test-Path "$repo\repro_runs\LATEST_SNNA_FULL.txt") { Get-Content "$repo\repro_runs\LATEST_SNNA_FULL.txt" -Raw }
Write-Host "---FULL RUNS---"
Get-ChildItem "$repo\repro_runs" -Directory -Filter 'full_*' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 10 FullName,LastWriteTime |
    Format-List
