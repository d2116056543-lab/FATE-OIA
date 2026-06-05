param(
  [string]$Config = "configs\fate_oia_train_360x640_egcaf_oia_v1.yaml",
  [int]$Epochs = 28,
  [int]$BatchSize = 4,
  [int]$GradAccum = 8,
  [int]$FallbackBatchSize1 = 3,
  [int]$FallbackGradAccum1 = 11,
  [int]$FallbackBatchSize2 = 2,
  [int]$FallbackGradAccum2 = 16,
  [string]$Device = "cuda",
  [string]$BDD100KRoot = "E:\sbw\BDD100K",
  [string]$BDDOIARoot = "E:\sbw\BDD-OIA",
  [string]$PretrainedWeights = "ckp/reference/dino_deitsmall8_pretrain.pth",
  [string]$CheckpointKey = "teacher",
  [switch]$NoFeatureCache,
  [switch]$TestOnly,
  [switch]$GoalMode,
  [switch]$RequireReviewPass,
  [int]$PrintEvery = 200
)
$ErrorActionPreference = "Stop"
$py = "E:\Anaconda\envs\sbw39\python.exe"
Write-Host "EG-CAF foreground supervisor starting"
Write-Host "Config=$Config Epochs=$Epochs BatchSize=$BatchSize GradAccum=$GradAccum Device=$Device"
Write-Host "PretrainedWeights=$PretrainedWeights CheckpointKey=$CheckpointKey"
& $py -m fate_oia.engine.supervise_egcaf_oia_foreground `
  --config $Config `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradAccum `
  --fallback_batch_size1 $FallbackBatchSize1 `
  --fallback_gradient_accumulation_steps1 $FallbackGradAccum1 `
  --fallback_batch_size2 $FallbackBatchSize2 `
  --fallback_gradient_accumulation_steps2 $FallbackGradAccum2 `
  --device $Device `
  --output_dir ".background_runs\egcaf_oia_v1_1_full_28" `
  --data_root "$BDDOIARoot\data" `
  --raw_root "$BDDOIARoot" `
  --bdd100k_root "$BDD100KRoot" `
  --pretrained_weights "$PretrainedWeights" `
  --checkpoint_key "$CheckpointKey" `
  --print_every $PrintEvery `
  --require_review_pass `
  --no_feature_cache `
  --test_only `
  --goal_mode
exit $LASTEXITCODE
