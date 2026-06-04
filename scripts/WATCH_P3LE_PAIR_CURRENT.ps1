param(
  [string]$RepoRoot = "E:\sbw\FATE_Drive\fate_oia_p3le_pair_oia_v1_worktree",
  [string]$RunDir = "",
  [int]$IntervalSeconds = 60
)
$ErrorActionPreference = "Continue"
Set-Location $RepoRoot
function Get-LatestRunDir {
  if ($RunDir -and (Test-Path $RunDir)) { return $RunDir }
  $root = Join-Path $RepoRoot ".background_runs"
  $latest = Get-ChildItem $root -Directory | Where-Object { $_.Name -like "p3le_pair_oia_v1_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $latest) { return $null }
  $train = Get-ChildItem $latest.FullName -Directory | Where-Object { $_.Name -like "train_*" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($train) { return $train.FullName }
  return $latest.FullName
}
function Print-Status {
  $dir = Get-LatestRunDir
  $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Write-Host "`n=== P3LE WATCH $now ==="
  if (-not $dir) { Write-Host "No run directory found."; return }
  Write-Host "RunDir: $dir"
  $metrics = Join-Path $dir "metrics_summary.jsonl"
  if (Test-Path $metrics) {
    $code = @"
import json, pathlib
p = pathlib.Path(r'$metrics')
rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
print('CompletedEpochs:', len(rows))
if rows:
    best = max(rows, key=lambda r: r.get('joint_test_score', -1))
    cur = rows[-1]
    for tag, r in [('Latest', cur), ('Best', best)]:
        m = r.get('test_metrics', {})
        print(f"{tag}: epoch={r.get('epoch')} joint={r.get('joint_test_score'):.6f} Act_mF1={m.get('Act_mF1'):.6f} Act_oF1={m.get('Act_oF1'):.6f} Exp_mF1={m.get('Exp_mF1'):.6f} Exp_oF1={m.get('Exp_oF1'):.6f} Exp_mAP={m.get('Exp_mAP'):.6f} train_loss={r.get('train_loss'):.6f} test_loss={r.get('test_loss'):.6f} lr={r.get('lr'):.9f}")
"@
    $code | E:\Anaconda\envs\sbw39\python.exe -
  } else {
    Write-Host "No metrics_summary.jsonl yet. Training may still be in epoch 0."
  }
  $latestEpoch = Get-ChildItem $dir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "epoch_*" } | Sort-Object Name -Descending | Select-Object -First 1
  if ($latestEpoch) {
    $lossFile = Join-Path $latestEpoch.FullName "loss_components.jsonl"
    if (Test-Path $lossFile) {
      $code2 = @"
import json, pathlib
p = pathlib.Path(r'$lossFile')
lines = [x for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
if lines:
    r = json.loads(lines[-1])
    print(f"LatestBatch: epoch={r.get('epoch')} step={r.get('step')} loss={r.get('loss'):.6f} action={r.get('action_loss'):.6f} reason={r.get('reason_loss'):.6f} pcgrad_conflicts={r.get('pcgrad_conflict_count')} batch={r.get('batch_size')} lr={r.get('lr'):.9f}")
"@
      $code2 | E:\Anaconda\envs\sbw39\python.exe -
    }
  }
  $needle = "p3le_pair|train_p3le|supervise_p3le"
  $procCount = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $needle }).Count
  Write-Host "P3LEProcesses: $procCount"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
}
while ($true) {
  Print-Status
  Start-Sleep -Seconds $IntervalSeconds
}
