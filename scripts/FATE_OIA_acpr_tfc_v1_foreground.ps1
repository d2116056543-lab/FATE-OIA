param(
  [int]$Epochs = 14,
  [int]$BatchSize = 4,
  [int]$GradAccum = 8,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
$Python = "E:\Anaconda\envs\sbw39\python.exe"
if (!(Test-Path $Python)) { $Python = "python" }
Write-Host "Running ACPR-TFC gates..."
& $Python -m fate_oia.engine.audit_tfc_gates --config configs\fate_oia_train_360x640_acpr_tfc_v1.yaml --mode all --device $Device --batch_size $BatchSize --write_review_pass
if ($LASTEXITCODE -ne 0) { throw "TFC audit gates failed" }
$reviewPath = ".review\acpr_tfc_v1_REVIEW_PASS.json"
if (!(Test-Path $reviewPath)) { throw "Missing review pass" }
$review = Get-Content -LiteralPath $reviewPath -Raw | ConvertFrom-Json
if (-not $review.review_pass) { throw "TFC review_pass=false" }
Write-Host "Starting ACPR-TFC foreground training..."
$trainArgs = @(
  "-u", "-m", "fate_oia.engine.train_acpr_tfc_oia",
  "--config", "configs\fate_oia_train_360x640_acpr_tfc_v1.yaml",
  "--output_dir", ".background_runs\acpr_tfc_v1_full",
  "--epochs", "$Epochs",
  "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum",
  "--num_workers", "4",
  "--device", "$Device"
)
if ($RequireReviewPass) {
  $trainArgs += "--require_review_pass"
}
& $Python @trainArgs
exit $LASTEXITCODE
