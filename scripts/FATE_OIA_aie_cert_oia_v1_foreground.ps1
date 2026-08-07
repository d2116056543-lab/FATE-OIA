param([int]$Epochs=16,[string]$Device='cuda',[switch]$RequireReviewPass)
$ErrorActionPreference='Stop'; $python='E:\Anaconda\envs\sbw39\python.exe'
$args=@('-u','-m','fate_oia.engine.supervise_aie_cert_oia_foreground','--config','configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml','--output-dir','.background_runs\aie_cert_oia_v1_full','--epochs',$Epochs,'--device',$Device)
if($RequireReviewPass){$args += '--require-review-pass'}
& $python @args
exit $LASTEXITCODE
