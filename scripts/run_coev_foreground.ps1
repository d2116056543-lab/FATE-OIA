param([string]$Config="configs/coev_oia_v1.yaml",[string]$Resume="")
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Config)) { throw "Missing config: $Config" }
if (-not (Test-Path -LiteralPath ".review\coev_v1\FULL_TRAIN_READY.json")) { throw "Missing FULL_TRAIN_READY" }
$argsList = @("-u","-m","fate_oia.utils.coev_supervisor","--config",$Config)
if ($Resume) { $argsList += @("--resume",$Resume) }
& python @argsList
$code=$LASTEXITCODE
exit $code
