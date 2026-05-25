param(
  [int]$MaxTrainSamples = 0,
  [int]$MaxValSamples = 0,
  [string]$OutputDir = ".background_runs\fate_oia_stage0_snna25"
)
$py = "E:\Anaconda\envs\sbw39\python.exe"
& $py -m fate_oia.engine.train_snna25 --output_dir $OutputDir --epochs 1 --max_train_samples $MaxTrainSamples --max_val_samples $MaxValSamples
