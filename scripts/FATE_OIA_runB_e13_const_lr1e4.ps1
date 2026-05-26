$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $Here "FATE_OIA_controlled_resume.ps1") `
  -RunName "runB_e13_const_lr1e4_keep070_cf0" `
  -Scheduler "none" `
  -Lr "0.0001" `
  -KeepRatio "0.70" `
  -LossCounterfactual "0" `
  -LabelCorrelation "none" `
  -TaskBalance "none" `
  @args
