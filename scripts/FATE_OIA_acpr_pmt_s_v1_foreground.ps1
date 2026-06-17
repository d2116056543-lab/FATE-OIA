param(
  [int]$Epochs = 18,
  [int]$BatchSize = 6,
  [int]$GradAccum = 5,
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
Set-Location E:\sbw\FATE_Drive\fate_oia_acpr_pmt_s_v1_worktree
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_acpr_pmt_s_foreground `
  --config configs\fate_oia_train_360x640_acpr_pmt_s_v1.yaml `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --device $Device `
  --output_dir .background_runs\acpr_pmt_s_v1_360x640_testonly_18e `
  --require_review_pass:$RequireReviewPass
