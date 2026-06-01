$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "E:\sbw\FATE_Drive\fate_oia_sure_oia_v2_worktree"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$out = ".background_runs\sure_oia_v2_direct_image_$ts"
E:\Anaconda\envs\sbw39\python.exe -m fate_oia.engine.supervise_sure_oia_foreground `
  --config configs\fate_oia_train_360x640_sure_oia_v2.yaml `
  --output_dir $out `
  --epochs 24 `
  --eval_splits test `
  --initial_batch_size 4 `
  --fallback_batch_sizes 3,2 `
  --python E:\Anaconda\envs\sbw39\python.exe
