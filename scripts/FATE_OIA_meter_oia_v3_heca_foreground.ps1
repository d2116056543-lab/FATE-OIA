param(
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [int]$NumWorkers = 4,
  [string]$Device = "cuda",
  [string]$OutputDir = ".background_runs\meter_oia_v3_heca_full",
  [string]$ReviewPass = ".background_runs\meter_oia_v3_heca_preflight\REVIEW_PASS_METER_OIA_V3_HECA.json",
  [string]$PilotPass = ".background_runs\meter_oia_v3_heca_pilot\HECA_PILOT_PASS.json",
  [string]$GateCPass = ".background_runs\meter_oia_v3_heca_pilot\HECA_GATE_C.json"
)

$ErrorActionPreference = "Stop"
& E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_meter_oia_v3_heca_foreground `
  --config configs\fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml `
  --output_dir $OutputDir --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum --num_workers $NumWorkers `
  --device $Device --review_pass $ReviewPass --pilot_pass $PilotPass `
  --gate_c_pass $GateCPass
if ($LASTEXITCODE -ne 0) { throw "HECA foreground training exited with code $LASTEXITCODE" }
