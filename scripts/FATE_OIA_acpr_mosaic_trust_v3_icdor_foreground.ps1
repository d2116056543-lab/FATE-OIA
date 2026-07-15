[CmdletBinding()]
param(
    [ValidateSet("audit", "profile", "pilot", "full")]
    [string]$Mode = "full",
    [string]$Config = "configs\fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml",
    [string]$Python = "E:\Anaconda\envs\damo39\python.exe",
    [string]$FinalRemediationPlan = "E:\sbw\FATE_Drive\Codex_MOSAIC_TRUST_V3_IC_DOR_Final_Remediation_20260714.md",
    [string]$AuditAddendum = "E:\sbw\FATE_Drive\MOSAIC_TRUST_V3_IC_DOR_Final_Audit_Addendum_20260714.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
$branch = (git branch --show-current).Trim()
if ($branch -ne "acpr_mosaic_trust_v3_icdor_direct_image") { throw "Wrong IC-DOR branch: $branch" }
$head = (git rev-parse HEAD).Trim()
if (-not $head) { throw "Cannot resolve current Git HEAD" }

function Get-Sha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required file is missing: $Path" }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Assert-ReviewPass {
    param([string]$ReviewPath, [string]$RuntimePath)
    if (-not (Test-Path -LiteralPath $ReviewPath)) { throw "REVIEW_PASS is missing; launch is forbidden" }
    $review = Get-Content -LiteralPath $ReviewPath -Raw | ConvertFrom-Json
    if ($review.status -ne "PASS") { throw "REVIEW_PASS status is not PASS" }
    if ($review.target_head -ne $head) { throw "REVIEW_PASS target_head does not bind current HEAD" }
    $configHash = Get-Sha256 $Config
    if ($review.resolved_config_sha256 -ne $configHash) { throw "REVIEW_PASS config hash mismatch" }
    if ($RuntimePath) {
        $runtimeHash = Get-Sha256 $RuntimePath
        if ($review.runtime_selection_sha256 -ne $runtimeHash) { throw "REVIEW_PASS runtime hash mismatch" }
    }
    foreach ($evidenceName in @("pilot_gate", "factor_certificate", "edge_admission", "final_remediation_plan", "audit_addendum")) {
        $property = $review.evidence_files.PSObject.Properties[$evidenceName]
        if ($null -eq $property) { throw "REVIEW_PASS $evidenceName binding is missing" }
        $binding = $property.Value
        if (-not $binding.path -or -not $binding.sha256) { throw "REVIEW_PASS $evidenceName binding is incomplete" }
        $actualHash = Get-Sha256 ([string]$binding.path)
        if ($actualHash -ne $binding.sha256) { throw "REVIEW_PASS $evidenceName evidence hash mismatch" }
    }
    foreach ($property in $review.gates.PSObject.Properties) {
        if ($property.Value -ne "PASS") { throw "REVIEW_PASS gate $($property.Name) is not PASS" }
    }
}

function Invoke-ForegroundPython {
    param([string[]]$Arguments)
    & $Python -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Foreground Python exited with code $LASTEXITCODE" }
}

$reviewPath = ".review\acpr_mosaic_trust_v3_icdor_REVIEW_PASS.json"
$runtimePath = ".review\mosaic_icdor_runtime_selection.json"
$pilotDir = ".background_runs\acpr_mosaic_trust_v3_icdor_pilot"
$pilotGate = Join-Path $pilotDir "pilot_gate.json"

switch ($Mode) {
    "audit" {
        Invoke-ForegroundPython @(
            "-m", "fate_oia.engine.audit_acpr_mosaic_trust_icdor",
            "--config", $Config,
            "--worktree_root", ".",
            "--output_dir", ".review",
            "--fail_closed"
        )
    }
    "profile" {
        Invoke-ForegroundPython @(
            "-m", "fate_oia.engine.profile_acpr_mosaic_trust_icdor",
            "--config", $Config,
            "--output", $runtimePath,
            "--device", "cuda"
        )
    }
    "pilot" {
        if (-not (Test-Path -LiteralPath $runtimePath)) { throw "Real runtime selection is missing" }
        Invoke-ForegroundPython @(
            "-m", "fate_oia.engine.train_acpr_mosaic_trust_icdor",
            "--config", $Config,
            "--output_dir", $pilotDir,
            "--runtime_selection", $runtimePath,
            "--pilot",
            "--epochs", "6",
            "--max_train_samples", "2048",
            "--max_audit_samples", "512",
            "--max_calib_samples", "512",
            "--max_test_samples", "512",
            "--seed", "20260713",
            "--device", "cuda"
        )
        Invoke-ForegroundPython @(
            "-m", "fate_oia.engine.audit_acpr_mosaic_trust_icdor",
            "--config", $Config,
            "--worktree_root", ".",
            "--output_dir", ".review",
            "--runtime_selection", $runtimePath,
            "--pilot_gate", $pilotGate,
            "--final_remediation_plan", $FinalRemediationPlan,
            "--audit_addendum", $AuditAddendum,
            "--write_review_pass",
            "--fail_closed",
            "--device", "cuda"
        )
    }
    "full" {
        Assert-ReviewPass -ReviewPath $reviewPath -RuntimePath $runtimePath
        Invoke-ForegroundPython @(
            "-m", "fate_oia.engine.train_acpr_mosaic_trust_icdor",
            "--config", $Config,
            "--output_dir", ".background_runs\acpr_mosaic_trust_v3_icdor_full",
            "--runtime_selection", $runtimePath,
            "--review_pass", $reviewPath,
            "--require_review_pass",
            "--epochs", "12",
            "--seed", "20260713",
            "--device", "cuda"
        )
    }
}

