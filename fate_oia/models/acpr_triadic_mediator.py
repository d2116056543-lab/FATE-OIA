from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
import yaml


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


class ACPRTriadicMediator(nn.Module):
    # PMT-S invariant: predicate-only action delta is impossible; action deltas require action-reason-predicate support.
    def __init__(self, action_dim=4, reason_dim=21, num_predicates=32, rank=8, max_action_delta=0.10, grammar_path="configs/acpr_reason_predicate_grammar.yaml", action_predicate_grammar_path="configs/acpr_pmt_action_predicate_grammar.yaml", scene_predicate_path="configs/acpr_scene_predicates.yaml"):
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_predicates = int(num_predicates)
        self.max_action_delta = float(max_action_delta)
        self.action_reason_weight = nn.Parameter(torch.zeros(action_dim, reason_dim))
        self.delta_scale = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.randn(action_dim, rank) * 0.01)
        self.R = nn.Parameter(torch.randn(reason_dim, rank) * 0.01)
        self.P = nn.Parameter(torch.randn(num_predicates, rank) * 0.01)

        predicate_names = [str(i) for i in range(num_predicates)]
        if Path(scene_predicate_path).exists():
            scene_data = yaml.safe_load(Path(scene_predicate_path).read_text(encoding="utf-8")) or {}
            for item in scene_data.get("predicates", []):
                pid = int(item.get("id", len(predicate_names)))
                if 0 <= pid < num_predicates:
                    predicate_names[pid] = str(item.get("name", pid))
        pred_lookup = {_norm(name): i for i, name in enumerate(predicate_names)}
        aliases = {
            "greenlight": "trafficlightgreen",
            "redlight": "trafficlightred",
            "stopsign": "stopsignpresent",
            "frontvehicle": "frontvehicleclose",
            "drivablefront": "drivablecenter",
            "laneclear": "roadclear",
            "leftlanemarking": "laneleftavailable",
            "rightlanemarking": "lanerightavailable",
            "leftfreespace": "openleftgap",
            "rightfreespace": "openrightgap",
            "laneleft": "laneleftavailable",
            "laneright": "lanerightavailable",
            "leftvehicle": "vehicleleft",
            "rightvehicle": "vehicleright",
            "frontobstacle": "obstaclefront",
            "leftobstacle": "vehicleleft",
            "rightobstacle": "vehicleright",
            "leftblocked": "vehicleleft",
            "rightblocked": "vehicleright",
            "speedlimit": "trafficsignvisible",
        }

        def resolve(names: list[str]) -> list[int]:
            ids: list[int] = []
            for name in names or []:
                n = _norm(name)
                n = aliases.get(n, n)
                if n in pred_lookup:
                    ids.append(pred_lookup[n])
                    continue
                for key, idx in pred_lookup.items():
                    if n and (n in key or key in n):
                        ids.append(idx)
            return sorted(set(i for i in ids if 0 <= i < num_predicates))

        ar = torch.zeros(action_dim, reason_dim)
        ap = torch.zeros(action_dim, num_predicates)
        ac = torch.zeros(action_dim, num_predicates)
        rp = torch.zeros(reason_dim, num_predicates)
        rc = torch.zeros(reason_dim, num_predicates)

        action_name_to_id: dict[str, int] = {}
        if Path(action_predicate_grammar_path).exists():
            data = yaml.safe_load(Path(action_predicate_grammar_path).read_text(encoding="utf-8")) or {}
            for action_name, item in data.get("actions", {}).items():
                aid = int(item.get("id", 0))
                if 0 <= aid < action_dim:
                    action_name_to_id[_norm(action_name)] = aid
                    for rid in item.get("compatible_reasons", []):
                        if 0 <= int(rid) < reason_dim:
                            ar[aid, int(rid)] = 1
                    for pid in resolve(item.get("support_predicates", [])):
                        ap[aid, pid] = 1
                    for pid in resolve(item.get("contradict_predicates", [])):
                        ac[aid, pid] = 1

        if Path(grammar_path).exists():
            grammar = yaml.safe_load(Path(grammar_path).read_text(encoding="utf-8")) or {}
            for action_key, item in grammar.get("actions", {}).items():
                aid = int(action_key)
                action_name_to_id[_norm(item.get("name", action_key))] = aid
            for reason_key, item in grammar.get("reasons", {}).items():
                rid = int(reason_key)
                if not (0 <= rid < reason_dim):
                    continue
                pos_ids = resolve(item.get("positive_predicates", []))
                neg_ids = resolve(item.get("contradictory_predicates", []))
                for pid in pos_ids:
                    rp[rid, pid] = 1
                for pid in neg_ids:
                    rc[rid, pid] = 1
                for act in item.get("compatible_actions", []):
                    aid = action_name_to_id.get(_norm(act))
                    if aid is not None and 0 <= aid < action_dim:
                        ar[aid, rid] = 1
                        for pid in pos_ids:
                            ap[aid, pid] = 1
                        for pid in neg_ids:
                            ac[aid, pid] = 1

        # Conservative fallbacks for rows with no declared support. These keep
        # tensors finite without returning to all-one or modulo placeholder masks.
        for a in range(action_dim):
            if ap[a].sum() <= 0:
                ap[a, min(a, num_predicates - 1)] = 1
            if ar[a].sum() <= 0:
                ar[a].fill_(1.0)
        for r in range(reason_dim):
            if rp[r].sum() <= 0:
                rp[r, min(r, num_predicates - 1)] = 1

        self.register_buffer("action_reason_compat_mask", ar)
        self.register_buffer("action_predicate_support_mask", ap)
        self.register_buffer("action_predicate_contradict_mask", ac)
        self.register_buffer("reason_predicate_positive_mask", rp)
        self.register_buffer("reason_predicate_contradictory_mask", rc)

    def forward(self, action_visual_logits, action_reason_logits, reason_logits, predicate_probs, predicate_tokens=None):
        e = predicate_probs
        pos_mask = self.reason_predicate_positive_mask.to(e.device, e.dtype)
        action_mask = self.action_predicate_support_mask.to(e.device, e.dtype)
        rp = (e @ pos_mask.t()) / pos_mask.sum(-1).clamp_min(1.0)
        ap = (e @ action_mask.t()) / action_mask.sum(-1).clamp_min(1.0)
        reason_conf = torch.sigmoid(reason_logits)
        support = self.action_reason_compat_mask.to(e.device, e.dtype).view(1, self.action_dim, self.reason_dim) * rp.unsqueeze(1) * ap.unsqueeze(-1)
        cp = torch.einsum("ak,rk,pk,bp->bar", self.A, self.R, self.P, e) / max(float(self.A.shape[-1]), 1.0)
        support = (support + 0.01 * cp).clamp(0.0, 1.0)
        chain = (
            self.action_reason_compat_mask.to(e.device, e.dtype).view(1, self.action_dim, self.reason_dim, 1)
            * action_mask.view(1, self.action_dim, 1, self.num_predicates)
            * pos_mask.view(1, 1, self.reason_dim, self.num_predicates)
            * e.view(e.shape[0], 1, 1, self.num_predicates)
            * reason_conf.view(e.shape[0], 1, self.reason_dim, 1)
        )
        effective_weight = self.action_reason_weight.view(1, self.action_dim, self.reason_dim) + cp
        raw_delta = (reason_conf.unsqueeze(1) * support * effective_weight).sum(-1)
        raw_delta = raw_delta + 0.01 * (reason_conf.unsqueeze(1) * support).sum(-1)
        delta = self.max_action_delta * torch.tanh(self.delta_scale * raw_delta)
        top_chain = chain.detach().flatten(1).topk(k=min(8, chain.shape[1] * chain.shape[2] * chain.shape[3]), dim=1).indices
        stats = {
            "triadic_delta_abs_mean": float(delta.detach().abs().mean().cpu()),
            "triadic_support_mean": float(support.detach().mean().cpu()),
            "triadic_action_predicate_support_density": float((self.action_predicate_support_mask > 0).float().mean().detach().cpu()),
            "triadic_reason_predicate_support_density": float((self.reason_predicate_positive_mask > 0).float().mean().detach().cpu()),
        }
        return {
            "action_reason_logits_triadic": action_reason_logits + delta,
            "triadic_action_delta": delta,
            "triadic_chain_score": chain,
            "triadic_reason_support": support,
            "triadic_predicate_support": action_mask.view(1, self.action_dim, self.num_predicates) * e.unsqueeze(1),
            "triadic_top_chain_indices": top_chain,
            "triadic_stats": stats,
        }
