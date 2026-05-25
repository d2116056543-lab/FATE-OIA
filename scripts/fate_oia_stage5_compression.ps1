param(
  [int]$MaxTrainSamples = 0,
  [int]$MaxValSamples = 0,
  [string]$OutputDir = ".background_runs\fate_oia_stage5_compression"
)
$py = "E:\Anaconda\envs\sbw39\python.exe"
& $py -m fate_oia.engine.train_fate_oia --output_dir $OutputDir --epochs 1 --max_train_samples $MaxTrainSamples --max_val_samples $MaxValSamples --image_height 360 --image_width 640 --batch_size 1 --token_compression keep_merge --compression_start_epoch 0 --compression_keep_ratio_start 0.75 --compression_keep_ratio_final 0.60 --num_summary_tokens 1
