param([string]$Config='configs/fate_oia_train_360x640_lens_oia_v1.yaml',[string]$OutputDir='.background_runs/lens_oia_v1_full',[string]$Device='cuda')
$ErrorActionPreference='Stop'
& E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_lens_oia_foreground --config $Config --output-dir $OutputDir --run-kind full --epochs 14 --device $Device
