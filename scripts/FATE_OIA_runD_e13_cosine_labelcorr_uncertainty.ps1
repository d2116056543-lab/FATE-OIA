$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here "FATE_OIA_controlled_resume.ps1") `
  -RunName "runD_e13_cosine_labelcorr_uncertainty" `
  -Scheduler "cosine" `
  -Lr "0.0001" `
  -MinLr "0.00001" `
  -KeepRatio "0.70" `
  -LossCounterfactual "0" `
  -LabelCorrelation "self_attn" `
  -TaskBalance "uncertainty" `
  @args
