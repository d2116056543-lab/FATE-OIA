$ErrorActionPreference = 'Stop'
$python = 'E:\Anaconda\envs\sbw39\python.exe'
Get-Content 'E:\sbw\FATE_Drive\task_plan.md' -Raw | Out-Null
Get-Content 'E:\sbw\FATE_Drive\findings.md' -Raw | Out-Null
Get-Content 'E:\sbw\FATE_Drive\progress.md' -Raw | Out-Null
$files = Get-ChildItem fate_oia -Recurse -Filter 'aie_cert_*.py' | ForEach-Object FullName
& $python -m py_compile @files
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m fate_oia.engine.profile_aie_cert_oia --config configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml --output .review\aie_cert_oia_v1\AIE_CERT_RUNTIME_PROFILE.json --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m fate_oia.engine.audit_aie_cert_oia_implementation --config configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml --device cuda --write-review-pass
exit $LASTEXITCODE
