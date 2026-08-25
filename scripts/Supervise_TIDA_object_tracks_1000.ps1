$ErrorActionPreference = "Stop"
$worktree = "E:\sbw\FATE_Drive\fate_oia_tida_relational_flow_v8_10k_worktree"
$partial = "F:\FATE_Drive_runs\tida_object_tracks_1000_calib512_test512.pt.partial"
$final = "F:\FATE_Drive_runs\tida_object_tracks_1000_calib512_test512.pt"
$python = "E:\Anaconda\envs\sbw39\python.exe"
$log = "F:\FATE_Drive_runs\tida_object_tracks_1000_supervisor.jsonl"
Set-Location $worktree

function Get-SavedCount {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return 0 }
  $value = & $python -c "import torch;p=torch.load(r'$Path',map_location='cpu',weights_only=True);print(len(p['file_names']))"
  if ($LASTEXITCODE -ne 0) { throw "failed to read object-track checkpoint" }
  return [int]$value
}

$stalled = 0
for ($attempt = 1; $attempt -le 100; $attempt++) {
  if (Test-Path $final) { exit 0 }
  $before = Get-SavedCount $partial
  & "$worktree\scripts\Extract_TIDA_object_tracks_1000.ps1"
  $code = $LASTEXITCODE
  if (Test-Path $final) { exit 0 }
  $after = Get-SavedCount $partial
  @{attempt=$attempt; exit_code=$code; before=$before; after=$after; timestamp=(Get-Date).ToString("o")} `
    | ConvertTo-Json -Compress | Add-Content -Path $log -Encoding UTF8
  if ($after -le $before) { $stalled++ } else { $stalled = 0 }
  if ($stalled -ge 2) { throw "object-track extraction made no progress twice at count $after" }
  Start-Sleep -Seconds 2
}
throw "object-track extraction exceeded restart budget"
