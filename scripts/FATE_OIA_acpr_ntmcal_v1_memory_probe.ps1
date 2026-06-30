$ErrorActionPreference = "Stop"
$out = ".background_runs\acpr_ntmcal_v1_memory_probe"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Remove-Item -Force (Join-Path $out "memory_probe.json") -ErrorAction SilentlyContinue
$probePy = Join-Path $out "probe_runner.py"
@"
import json, time, sys, os, torch
sys.path.insert(0, os.getcwd())
from pathlib import Path
from fate_oia.engine.train_acpr_ntmcal_oia import load_config, make_loader, build_model, optimizer_for
from fate_oia.losses.acpr_ntmcal_losses import acpr_ntmcal_loss_bundle

cfg = load_config('configs/fate_oia_train_360x640_acpr_ntmcal_v1.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
candidates = [(8,4), (6,5), (4,8)]
results = []
for batch, accum in candidates:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg, device)
    optim = optimizer_for(model, cfg)
    loader = make_loader(cfg, 'train', batch, batch * 2, True, 0)
    max_alloc = max_reserved = 0.0
    ok = True
    msg = ''
    start = time.time()
    try:
        optim.zero_grad(set_to_none=True)
        for step, data in enumerate(loader, start=1):
            if step > 2:
                break
            images = data['image'].to(device)
            action = data['action'].to(device)
            reason = data['reason'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                out = model(images, epoch=0, split='train', reason_labels=reason, file_names=data['file_name'])
                out['_atom_encoder'] = model.atom_encoder
                out['_predicate_specs'] = model.predicate_bank.specs
                out['_pair_memory'] = model.pair_memory
                loss, stats = acpr_ntmcal_loss_bundle(out, action, reason, 0, cfg)
                loss = loss / accum
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
                max_alloc = max(max_alloc, torch.cuda.max_memory_allocated() / (1024**3))
                max_reserved = max(max_reserved, torch.cuda.max_memory_reserved() / (1024**3))
        optim.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    except Exception as exc:
        ok = False
        msg = repr(exc)
    elapsed = time.time() - start
    accepted = ok and max_reserved <= 42.0
    results.append({'batch': batch, 'accum': accum, 'ok': ok, 'accepted': accepted, 'step_time_sec': elapsed / 2.0, 'reserved_memory_gb': max_reserved, 'allocated_memory_gb': max_alloc, 'message': msg})
    del model, optim, loader
    torch.cuda.empty_cache() if device.type == 'cuda' else None

accepted = [r for r in results if r['accepted']]
selected = min(accepted, key=lambda r: r['step_time_sec']) if accepted else None
summary = {'candidates': results, 'selected': selected, 'bf16': device.type == 'cuda'}
Path('.background_runs/acpr_ntmcal_v1_memory_probe/memory_probe.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
if selected is None:
    raise SystemExit(1)
"@ | Set-Content -Encoding UTF8 $probePy
E:\Anaconda\envs\sbw39\python.exe $probePy


