$ErrorActionPreference = 'Stop'
wsl.exe -d ADAPT-Ubuntu -- bash /mnt/e/sbw/SNNA_repro/SNNA/scripts_repro/diagnose_snna_nccl.sh
if ($LASTEXITCODE -ne 0) {
    throw "NCCL diagnosis failed with exit code $LASTEXITCODE"
}
