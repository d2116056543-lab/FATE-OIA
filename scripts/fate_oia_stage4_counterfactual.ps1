param(
  [int]$MaxTrainSamples = 0,
  [int]$MaxValSamples = 0,
  [string]$OutputDir = ".background_runs\fate_oia_stage4_counterfactual"
)
$py = "E:\Anaconda\envs\sbw39\python.exe"
& $py -m fate_oia.engine.train_fate_oia --output_dir $OutputDir --epochs 1 --max_train_samples $MaxTrainSamples --max_val_samples $MaxValSamples --image_height 360 --image_width 640 --batch_size 1 --loss_counterfactual 0.01 --counterfactual_topk_ratio 0.10
