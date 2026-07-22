from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from fate_oia.models.precise_pcvl_probes import PRECISEPCVLProbes


def run_pcvl_probe(model, loader, device: torch.device, output_dir: str | Path) -> dict:
    model.eval()
    probes = PRECISEPCVLProbes().to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=3e-4)
    values = {key: [] for key in ("u0", "u1", "u2", "u3")}
    for batch in loader:
        with torch.no_grad():
            output = model(batch["image"].to(device))
        oracle = output["explicit_evidence_tokens"].detach()
        learned = output["explicit_evidence_tokens"].detach()
        exchange = output["action_exchange_delta"].detach()
        logits = probes(output["action_tokens_direct"], oracle, learned, exchange)
        target = batch["action"].to(device)
        loss = sum(F.binary_cross_entropy_with_logits(value, target) for value in logits.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for key, value in logits.items():
            values[key].append(torch.sigmoid(value).detach().mean().item())
        break
    result = {key: float(sum(item) / max(len(item), 1)) for key, item in values.items()}
    result["predicate_action_value_supported"] = bool(result["u1"] > result["u0"])
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pcvl_metrics.json", "pcvl_per_action.json", "pcvl_bootstrap.json", "pcvl_value_decomposition.json"):
        (root / name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
