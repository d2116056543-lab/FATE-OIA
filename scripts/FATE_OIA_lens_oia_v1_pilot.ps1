param([string]$Config='configs/fate_oia_train_360x640_lens_oia_v1.yaml',[string]$OutputDir='.background_runs/lens_oia_v1_pilot',[string]$Device='cuda')
$ErrorActionPreference='Stop'
& E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.train_lens_oia --config $Config --output-dir $OutputDir --epochs 4 --max-train-samples 4096 --max-calib-samples 512 --max-test-samples 512 --device $Device
& E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.evaluate_lens_oia_pilot --run-dir $OutputDir
