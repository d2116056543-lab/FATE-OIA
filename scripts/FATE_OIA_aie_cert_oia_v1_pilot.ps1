param([int]$Epochs=3,[int]$MaxTrainSamples=4096,[int]$MaxCalibSamples=512,[int]$MaxAuditSamples=512,[int]$MaxTestSamples=512)
$ErrorActionPreference = 'Stop'; $python='E:\Anaconda\envs\sbw39\python.exe'; $review='.review\aie_cert_oia_v1'
if (-not (Test-Path "$review\REVIEW_PASS_AIE_CERT_OIA_V1.json")) { throw 'REVIEW_PASS missing' }
& $python -m fate_oia.engine.profile_aie_cert_oia --config configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml --output "$review\AIE_CERT_RUNTIME_PROFILE.json" --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -u -m fate_oia.engine.train_aie_cert_oia --config configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml --output-dir .background_runs\aie_cert_oia_v1_pilot --run-kind pilot --epochs $Epochs --max-train-samples $MaxTrainSamples --max-calib-samples $MaxCalibSamples --max-audit-samples $MaxAuditSamples --max-test-samples $MaxTestSamples --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m fate_oia.engine.evaluate_aie_cert_oia_pilot --pilot-dir .background_runs\aie_cert_oia_v1_pilot `
  --config configs\fate_oia_train_360x640_aie_cert_oia_v1.yaml --review-dir $review
exit $LASTEXITCODE
