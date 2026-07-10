param(
    [ValidateSet("pilot", "full")]
    [string]$Mode,
    [string]$Python = "E:\Anaconda\envs\sbw39\python.exe",
    [string]$Config = "configs\fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml",
    [string]$RuntimeSelection = ".review\mosaic_runtime_selection.json"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not (Test-Path -LiteralPath $Python)) { throw "Python runtime missing: $Python" }
if (-not (Test-Path -LiteralPath $Config)) { throw "MOSAIC config missing: $Config" }
if (-not (Test-Path -LiteralPath $RuntimeSelection)) { throw "Runtime selection missing: $RuntimeSelection" }

$runtime = Get-Content -LiteralPath $RuntimeSelection -Raw | ConvertFrom-Json
if ($runtime.pass -ne $true -or $runtime.quick_diagnostic -eq $true -or $runtime.stability_probe.pass -ne $true) {
    throw "Runtime selection is not a formal PASS with stability probe"
}
$batchSize = [int]$runtime.selected.batch_size
$gradAccum = [int]$runtime.selected.grad_accum
$numWorkers = [int]$runtime.selected.num_workers
$head = (git rev-parse HEAD).Trim()
$branch = (git branch --show-current).Trim()
if ($branch -ne "acpr_mosaic_ad_v1_direct_image") { throw "Wrong branch: $branch" }

function Invoke-ForegroundPython {
    param([string[]]$Arguments, [string]$ManifestDir)
    New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null
    $manifest = @{
        powershell_pid = $PID
        mode = $Mode
        python = $Python
        arguments = $Arguments
        git_head = $head
        foreground = $true
        started_at = (Get-Date).ToString("o")
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ManifestDir "foreground_command.json") -Encoding UTF8
    $PID | Set-Content -LiteralPath (Join-Path $ManifestDir "foreground_supervisor.pid") -Encoding ASCII
    & $Python -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Foreground Python exited with code $LASTEXITCODE" }
}

function Assert-NewRunDirectory {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to mix MOSAIC artifacts with an existing run directory: $Path"
    }
}

if ($Mode -eq "pilot") {
    $artifactSmokeDir = ".background_runs\acpr_mosaic_ad_v1_artifact_smoke"
    Assert-NewRunDirectory -Path $artifactSmokeDir
    $artifactArgs = @(
        "-m", "fate_oia.engine.train_acpr_mosaic_ad",
        "--config", $Config,
        "--output_dir", $artifactSmokeDir,
        "--runtime_selection", $RuntimeSelection,
        "--device", "cuda",
        "--epochs", "2",
        "--batch_size", "$batchSize",
        "--grad_accum", "$gradAccum",
        "--num_workers", "$numWorkers",
        "--max_train_samples", "512",
        "--max_calib_samples", "256",
        "--max_test_samples", "256",
        "--seed", "20260710"
    )
    Invoke-ForegroundPython -Arguments $artifactArgs -ManifestDir $artifactSmokeDir

    $pilotRoot = ".background_runs\acpr_mosaic_ad_v1_pilot"
    foreach ($seed in @(20260710, 20260711, 20260712)) {
        $seedDir = Join-Path $pilotRoot "seed_$seed"
        Assert-NewRunDirectory -Path $seedDir
        $trainArgs = @(
            "-m", "fate_oia.engine.train_acpr_mosaic_ad",
            "--config", $Config,
            "--output_dir", $seedDir,
            "--runtime_selection", $RuntimeSelection,
            "--device", "cuda",
            "--epochs", "8",
            "--batch_size", "$batchSize",
            "--grad_accum", "$gradAccum",
            "--num_workers", "$numWorkers",
            "--max_train_samples", "4096",
            "--max_calib_samples", "1024",
            "--max_test_samples", "1024",
            "--seed", "$seed"
        )
        Invoke-ForegroundPython -Arguments $trainArgs -ManifestDir $seedDir
        $visualArgs = @(
            "-m", "fate_oia.engine.export_mosaic_visual_audit",
            "--config", $Config,
            "--checkpoint", (Join-Path $seedDir "checkpoint_best_test_joint.pth"),
            "--output_dir", (Join-Path $seedDir "visual_audit"),
            "--device", "cuda",
            "--max_samples", "512",
            "--batch_size", "$batchSize",
            "--num_workers", "$numWorkers"
        )
        Invoke-ForegroundPython -Arguments $visualArgs -ManifestDir (Join-Path $seedDir "visual_audit")
    }
    $auditArgs = @(
        "-m", "fate_oia.engine.audit_acpr_mosaic_ad",
        "--config", $Config,
        "--output_dir", ".review",
        "--device", "cuda",
        "--artifact_smoke_dir", $artifactSmokeDir,
        "--pilot_dir", $pilotRoot,
        "--write_review_pass"
    )
    Invoke-ForegroundPython -Arguments $auditArgs -ManifestDir ".review"

    $remoteLine = git ls-remote github refs/heads/acpr_mosaic_ad_v1_direct_image
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($remoteLine)) { throw "Cannot verify GitHub branch after pilot" }
    $remoteHead = ($remoteLine -split "\s+")[0]
    if ($remoteHead -ne $head) { throw "GitHub HEAD $remoteHead differs from pilot HEAD $head" }
    $driveRoot = Split-Path -Parent $repo
    $verify = Join-Path $driveRoot "_verify_acpr_mosaic_ad_v1"
    $resolvedRoot = [IO.Path]::GetFullPath($driveRoot)
    $resolvedVerify = [IO.Path]::GetFullPath($verify)
    if (-not $resolvedVerify.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolvedVerify) -ne "_verify_acpr_mosaic_ad_v1") {
        throw "Unsafe fresh-clone verification path: $resolvedVerify"
    }
    if (Test-Path -LiteralPath $resolvedVerify) { Remove-Item -LiteralPath $resolvedVerify -Recurse -Force }
    git clone --branch acpr_mosaic_ad_v1_direct_image https://github.com/d2116056543-lab/FATE-OIA.git $resolvedVerify
    if ($LASTEXITCODE -ne 0) { throw "Fresh GitHub clone failed" }
    Push-Location $resolvedVerify
    try {
        & $Python -m compileall fate_oia
        if ($LASTEXITCODE -ne 0) { throw "Fresh clone compile failed" }
        $testFiles = Get-ChildItem -LiteralPath tests -Filter "test_mosaic_*.py" | ForEach-Object { $_.FullName }
        & $Python -m pytest @testFiles -q
        if ($LASTEXITCODE -ne 0) { throw "Fresh clone MOSAIC tests failed" }
        $freshHead = (git rev-parse HEAD).Trim()
    }
    finally { Pop-Location }
    if ($freshHead -ne $head) { throw "Fresh clone HEAD differs from pilot HEAD" }
    @{
        status = "PASS"
        local_head = $head
        remote_head = $remoteHead
        fresh_clone_head = $freshHead
        compile_pass = $true
        tests_pass = $true
        timestamp = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath ".review\github_sync_pass.json" -Encoding UTF8
    exit 0
}

$reviewPath = ".review\acpr_mosaic_ad_v1_REVIEW_PASS.json"
if (-not (Test-Path -LiteralPath $reviewPath)) { throw "REVIEW_PASS missing; full training forbidden" }
$review = Get-Content -LiteralPath $reviewPath -Raw | ConvertFrom-Json
if ($review.status -ne "PASS" -or $review.git_head -ne $head) { throw "REVIEW_PASS does not bind current HEAD" }
$configHash = (Get-FileHash -LiteralPath $Config -Algorithm SHA256).Hash.ToUpperInvariant()
$runtimeHash = (Get-FileHash -LiteralPath $RuntimeSelection -Algorithm SHA256).Hash.ToUpperInvariant()
if ($review.config_hash -ne $configHash) { throw "REVIEW_PASS config hash does not match requested config" }
if ($review.runtime_selection_hash -ne $runtimeHash) { throw "REVIEW_PASS runtime hash does not match requested runtime selection" }
$syncPath = ".review\github_sync_pass.json"
if (-not (Test-Path -LiteralPath $syncPath)) { throw "GitHub fresh-clone verification PASS is missing" }
$sync = Get-Content -LiteralPath $syncPath -Raw | ConvertFrom-Json
if ($sync.status -ne "PASS" -or $sync.local_head -ne $head -or $sync.remote_head -ne $head -or $sync.fresh_clone_head -ne $head) {
    throw "GitHub synchronization gate does not bind current HEAD"
}
$remoteLine = git ls-remote github refs/heads/acpr_mosaic_ad_v1_direct_image
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($remoteLine)) { throw "Cannot verify GitHub branch HEAD" }
$remoteHead = ($remoteLine -split "\s+")[0]
if ($remoteHead -ne $head) { throw "GitHub HEAD $remoteHead differs from local HEAD $head" }

$fullDir = ".background_runs\acpr_mosaic_ad_v1_full"
Assert-NewRunDirectory -Path $fullDir
$fullArgs = @(
    "-m", "fate_oia.engine.train_acpr_mosaic_ad",
    "--config", $Config,
    "--output_dir", $fullDir,
    "--runtime_selection", $RuntimeSelection,
    "--device", "cuda",
    "--epochs", "15",
    "--batch_size", "$batchSize",
    "--grad_accum", "$gradAccum",
    "--num_workers", "$numWorkers",
    "--seed", "20260710"
)
Invoke-ForegroundPython -Arguments $fullArgs -ManifestDir $fullDir
