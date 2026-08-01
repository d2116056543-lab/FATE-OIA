param(
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [int]$NumWorkers = 4,
  [string]$Device = "cuda",
  [string]$OutputDir = ".background_runs\meter_oia_v3_heca_pilot",
  [string]$PreflightDir = ".background_runs\meter_oia_v3_heca_preflight"
)

$ErrorActionPreference = "Stop"
$Python = "E:\Anaconda\envs\sbw39\python.exe"
$Config = "configs\fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml"
$TextEncoderPath = "artifacts\heca\frozen_bert_base_uncased"

$ExportArgs = @("-m", "fate_oia.engine.export_heca_ontology_prototypes", "--schema", "configs\meter_factor_schema.yaml", "--output_dir", "artifacts\heca", "--encoder_id", $TextEncoderPath)
& $Python @ExportArgs
if ($LASTEXITCODE -ne 0) { throw "HECA ontology export failed" }

$TauArgs = @("-m", "fate_oia.engine.prepare_heca_static_artifacts", "--config", $Config, "--output_dir", "artifacts\heca")
& $Python @TauArgs
if ($LASTEXITCODE -ne 0) { throw "HECA tau preparation failed" }

$AuditArgs = @("-m", "fate_oia.engine.audit_meter_oia_v3_heca", "--config", $Config, "--output_dir", $PreflightDir, "--write_review_pass")
& $Python @AuditArgs
if ($LASTEXITCODE -ne 0) { throw "HECA implementation audit failed" }

$TrainArgs = @(
  "-u", "-m", "fate_oia.engine.train_acpr_meter_oia",
  "--config", $Config, "--output_dir", $OutputDir, "--device", $Device,
  "--epochs", "4", "--batch_size", "$BatchSize",
  "--gradient_accumulation_steps", "$GradAccum", "--num_workers", "$NumWorkers",
  "--max_train_samples", "4096", "--max_audit_samples", "1024",
  "--max_calib_samples", "512", "--max_test_samples", "512",
  "--seed", "20260801", "--run_kind", "pilot", "--test_only",
  "--no_feature_cache", "--require_no_token_compression"
)
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "HECA four-epoch pilot failed" }

$GateArgs = @(
  "-m", "fate_oia.engine.evaluate_meter_oia_v3_heca_pilot",
  "--pilot_dir", $OutputDir,
  "--implementation_audit", "$PreflightDir\implementation_audit_METER_OIA_V3_HECA.json",
  "--ontology_manifest", "artifacts\heca\heca_ontology_manifest.json",
  "--tau_stats", "artifacts\heca\heca_tau_stats.json"
)
& $Python @GateArgs
if ($LASTEXITCODE -ne 0) { throw "HECA pilot gates A-G failed" }
