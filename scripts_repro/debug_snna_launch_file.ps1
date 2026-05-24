param(
    [string]$RunDir = 'E:\sbw\SNNA_repro\SNNA\repro_runs\full_20260522_183115'
)
$ErrorActionPreference = 'Stop'
$runWsl = $RunDir.Replace('E:\sbw\SNNA_repro\SNNA', '/mnt/e/sbw/SNNA_repro/SNNA').Replace('\','/')
$launch = "$runWsl/launch_full.sh"
wsl.exe -d ADAPT-Ubuntu -- bash -lc "echo LAUNCH=$launch; ls -l '$launch'; sed -n '1,20p' '$launch'; bash -n '$launch'; echo bash_n_exit=`$?"
if ($LASTEXITCODE -ne 0) { throw "debug launch failed $LASTEXITCODE" }
