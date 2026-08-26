$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts\FATE_OIA_tida_v9_4_primary5584_route.ps1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts\FATE_OIA_tida_v9_4_primary5584_utility.ps1
exit $LASTEXITCODE
