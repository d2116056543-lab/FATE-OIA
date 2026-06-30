param(
  [int]$Epochs = 18,
  [int]$BatchSize = 8,
  [int]$GradAccum = 4,
  [int]$NumWorkers = 8,
  [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\FATE_OIA_acpr_ntmcal_v1_memory_probe.ps1
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.audit_acpr_ntmcal_implementation --config configs\fate_oia_train_360x640_acpr_ntmcal_v1.yaml --output_dir .background_runs\acpr_ntmcal_v1_preflight --device $Device --write_review_pass
E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_acpr_ntmcal_oia --config configs\fate_oia_train_360x640_acpr_ntmcal_v1.yaml --output_dir .background_runs\acpr_ntmcal_v1_360x640_testprimary --epochs $Epochs --batch_size $BatchSize --gradient_accumulation_steps $GradAccum --num_workers $NumWorkers --device $Device --amp_dtype bf16 --test_only --no_feature_cache --token_compression none --require_review_pass .background_runs\acpr_ntmcal_v1_preflight\REVIEW_PASS_ACPR_NTMCAL_V1.txt
