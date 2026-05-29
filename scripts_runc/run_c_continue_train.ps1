param(
  [string]$OutputDir = "",
  [int]$Epochs = 24,
  [int]$BatchSize = 4,
  [int]$GradientAccumulationSteps = 8,
  [int]$MaxTrainSamples = 0,
  [int]$MaxValSamples = 0,
  [int]$MaxTestSamples = 0,
  [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path $PSScriptRoot -Parent
Set-Location $Repo
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $OutputDir = Join-Path $Repo "runc_outputs\run_c_continue_$stamp"
}
$DataRoot = "E:\sbw\FATE_Drive\fate_oia_worktree\dataset\BDD-OIA"
$RawRoot = "E:\sbw\FATE_Drive\fate_oia_worktree\raw_data\BDD-OIA"
$Pretrained = "E:\sbw\FATE_Drive\fate_oia_worktree\ckp\reference\dino_deitsmall8_pretrain.pth"
$GroundingCache = "E:\sbw\FATE_Drive\fate_oia_worktree\.background_runs\fate_oia_grounding_cache_20260525.jsonl"
$Resume = Join-Path $Repo "run_c_artifacts\checkpoint_best_test.pth"
foreach ($p in @($DataRoot, $RawRoot, $Pretrained, $GroundingCache, $Resume)) {
  if (-not (Test-Path $p)) { throw "Missing required Run C path: $p" }
}
Write-Host "Run C continue output: $OutputDir"
& "E:\Anaconda\envs\sbw39\python.exe" -m fate_oia.engine.train_fate_oia `
  --config "configs\fate_oia_train_360x640.yaml" `
  --data_root $DataRoot `
  --raw_root $RawRoot `
  --pretrained_weights $Pretrained `
  --grounding_cache_jsonl $GroundingCache `
  --reason_grounding_rules "configs\reason_grounding_rules.yaml" `
  --output_dir $OutputDir `
  --epochs $Epochs `
  --batch_size $BatchSize `
  --gradient_accumulation_steps $GradientAccumulationSteps `
  --lr 0.0001 `
  --scheduler cosine `
  --min_lr 0.00001 `
  --warmup_epochs 0 `
  --weight_decay 0.0001 `
  --loss asl `
  --asl_gamma_pos 0 `
  --asl_gamma_neg 4 `
  --asl_clip 0.05 `
  --loss_reason_to_action 0.1 `
  --loss_action_visual 0.05 `
  --loss_r2a_gt 0.1 `
  --loss_action_agree 0.01 `
  --no-include_fused_branch_loss `
  --loss_action_fused_aux 0 `
  --r2a_consistency_mode gt_and_agree `
  --fusion_mode learned_gate `
  --loss_gate_balance 0 `
  --loss_gate_entropy 0 `
  --loss_grounding 0.0001 `
  --grounding_mode both `
  --loss_counterfactual 0 `
  --counterfactual_eval `
  --counterfactual_start_epoch 0 `
  --counterfactual_topk_ratio 0.05 `
  --cf_mask_fill mean `
  --token_compression keep_merge `
  --compression_start_epoch 0 `
  --compression_warmup_epochs 0 `
  --compression_keep_ratio_start 0.70 `
  --compression_keep_ratio_final 0.70 `
  --token_score_mode norm `
  --num_summary_tokens 4 `
  --min_tokens 128 `
  --use_label_query `
  --use_reason_to_action `
  --label_correlation self_attn `
  --label_correlation_layers 1 `
  --label_correlation_heads 4 `
  --label_correlation_dropout 0.1 `
  --label_correlation_bias none `
  --task_balance none `
  --resume $Resume `
  --no-resume_optimizer `
  --resume_scheduler `
  --resume_strict `
  --best_selection_split test `
  --best_selection_metric joint_test_score `
  --render_explanation_text `
  --save_epoch_artifacts `
  --max_train_samples $MaxTrainSamples `
  --max_val_samples $MaxValSamples `
  --max_test_samples $MaxTestSamples `
  --num_workers 0 `
  --log_every 20 `
  --device $Device
exit $LASTEXITCODE
