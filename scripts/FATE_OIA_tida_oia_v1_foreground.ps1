param(
    [Parameter(Mandatory=$true)][string]$ClipManifest,
    [Parameter(Mandatory=$true)][string]$ImageCheckpoint,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$Config = "configs\fate_oia_train_tida_oia_v1_15f.yaml",
    [string]$ReviewReady = ".review\tida_oia_v1\FULL_TRAIN_READY_TIDA_OIA_V1.json",
    [string]$Python = "E:\Anaconda\envs\sbw39\python.exe",
    [int]$BatchSize = 2,
    [int]$GradAccum = 15,
    [int]$ContextChunkSize = 5,
    [int]$NumWorkers = 6,
    [string]$Device = "cuda",
    [string]$Resume = ""
)

$arguments = @(
    "-u", "-m", "fate_oia.engine.supervise_tida_oia_foreground",
    "--config", $Config,
    "--clip-manifest", $ClipManifest,
    "--image-checkpoint", $ImageCheckpoint,
    "--output-dir", $OutputDir,
    "--review-ready", $ReviewReady,
    "--python", $Python,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--gradient-accumulation-steps", $GradAccum,
    "--context-chunk-size", $ContextChunkSize,
    "--num-workers", $NumWorkers
)
if ($Resume) { $arguments += @("--resume", $Resume) }
& $Python @arguments
exit $LASTEXITCODE
