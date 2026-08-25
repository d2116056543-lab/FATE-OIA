param(
  [string]$OutputDir = 'F:\FATE_Drive_runs\tida_object_role_head_1000_v88'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
& 'E:\Anaconda\envs\sbw39\python.exe' -u -m fate_oia.engine.train_tida_object_role_head `
  --config 'configs\fate_oia_train_tida_object_intent_v8_4_10k.yaml' `
  --image_root 'E:\sbw\BDD100K\bdd100k_images\bdd100k\images\100k\train' `
  --label_root 'E:\sbw\BDD100K\bdd100k_labels\bdd100k\labels\100k\train' `
  --output_dir $OutputDir --max_images 1000 --epochs 5 --batch_size 6 --num_workers 4 --device cuda
exit $LASTEXITCODE
