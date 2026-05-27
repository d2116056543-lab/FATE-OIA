param(
  [string]$Root = "E:\sbw\FATE_Drive",
  [int]$BatchSize = 4,
  [int]$GradAccum = 8,
  [int]$S1Epochs = 25,
  [switch]$Smoke
)
$ErrorActionPreference = "Stop"
Set-Location "$Root\fate_oia_worktree"
$Py = "E:\Anaconda\envs\sbw39\python.exe"
& $Py -m fate_oia.engine.supervise_evis_oia `
  --root $Root `
  --fate_oia_dir "$Root\fate_oia_worktree" `
  --reference_run_c_dir "$Root\fate_oia_worktree\.background_runs\fate_oia_runC_e13_cosine_labelcorr_20260526_191938" `
  --batch_size $BatchSize `
  --grad_accum $GradAccum `
  --s1_epochs $S1Epochs `
  --allow_training `
  --smoke:$Smoke
exit $LASTEXITCODE
