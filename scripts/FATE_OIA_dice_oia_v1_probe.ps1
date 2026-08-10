param([string]$Config="configs\fate_oia_train_360x640_dice_oia_v1_probe.yaml",[Parameter(Mandatory=$true)][string]$BaseCheckpoint,[string]$OutputDir=".background_runs\dice_oia_v1_probe",[int]$Epochs=2,[string]$Device="cuda")
$ErrorActionPreference="Stop"
& E:\Anaconda\envs\sbw39\python.exe -u -m fate_oia.engine.supervise_dice_oia_probe --config $Config --base-checkpoint $BaseCheckpoint --output-dir $OutputDir --epochs $Epochs --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
