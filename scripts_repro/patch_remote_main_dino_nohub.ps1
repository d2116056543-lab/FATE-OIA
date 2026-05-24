$ErrorActionPreference = 'Stop'
Set-Location 'E:\sbw\SNNA_repro\SNNA'

$path = 'main_dino.py'
$text = Get-Content -LiteralPath $path -Raw
if ($text -match 'SNNA_ALLOW_XCIT_HUB') {
    Write-Host 'main_dino_nohub_xcit_already_present'
    exit 0
}

$new = @"
if os.environ.get("SNNA_ALLOW_XCIT_HUB", "0") == "1":
    try:
        xcit_archs = torch.hub.list("facebookresearch/xcit:main")
    except Exception as exc:
        print(f"Warning: unable to query XCiT torch.hub arch list ({exc}); "
              "continuing with local ViT/torchvision choices.")
        xcit_archs = []
else:
    xcit_archs = []
"@

$pattern = '(?s)try:\s*\r?\n\s*xcit_archs = torch\.hub\.list\("facebookresearch/xcit:main"\)\s*\r?\nexcept Exception as exc:\s*\r?\n\s*print\(f"Warning: unable to query XCiT torch\.hub arch list \(\{exc\}\); "\s*\r?\n\s*"continuing with local ViT/torchvision choices\."\)\s*\r?\n\s*xcit_archs = \[\]'

if ($text -notmatch $pattern) {
    throw 'Expected XCiT torch.hub block not found in main_dino.py'
}

$text = [regex]::Replace($text, $pattern, $new, 1)
Set-Content -LiteralPath $path -Value $text -Encoding UTF8
Write-Host 'main_dino_nohub_xcit_updated'
