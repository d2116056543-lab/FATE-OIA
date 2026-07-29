param(
  [int]$Epochs = 12,
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [int]$NumWorkers = 4,
  [string]$Device = "cuda",
  [string]$OutputDir = ".background_runs\acpr_meter_oia_v2_tesa_full",
  [string]$ReviewPass = ".background_runs\acpr_meter_oia_v2_tesa_preflight\REVIEW_PASS_METER_OIA_V2_TESA.txt",
  [string]$PilotPass = ".background_runs\acpr_meter_oia_v2_tesa_pilot\TESA_PILOT_PASS.json"
)

$ErrorActionPreference = "Stop"
& E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_acpr_meter_oia_foreground `
  --config configs\fate_oia_train_360x640_acpr_meter_oia_v2_tesa.yaml `
  --output_dir $OutputDir `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --num_workers $NumWorkers `
  --device $Device `
  --review_pass $ReviewPass `
  --pilot_pass $PilotPass
if ($LASTEXITCODE -ne 0) { throw "TESA foreground training exited with code $LASTEXITCODE" }
