$ErrorActionPreference = 'Stop'
Set-Location 'E:\sbw\SNNA_repro\SNNA'

$path = 'utils.py'
$text = Get-Content -LiteralPath $path -Raw
if ($text -match 'SNNA_DIST_BACKEND') {
    Write-Host 'utils_dist_backend_already_present'
    exit 0
}

$new = @"
    dist_backend = os.environ.get("SNNA_DIST_BACKEND", "nccl")
    dist.init_process_group(
        backend=dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
"@

$pattern = '(?s)    dist\.init_process_group\(\s*\r?\n        backend="nccl",\s*\r?\n        init_method=args\.dist_url,\s*\r?\n        world_size=args\.world_size,\s*\r?\n        rank=args\.rank,\s*\r?\n    \)'
if ($text -notmatch $pattern) {
    throw 'Expected nccl init_process_group block not found in utils.py'
}

$text = [regex]::Replace($text, $pattern, $new, 1)
Set-Content -LiteralPath $path -Value $text -Encoding UTF8
Write-Host 'utils_dist_backend_updated'
