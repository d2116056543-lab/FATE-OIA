from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor, nn


def _interp(t0: Tensor, t1: Tensor, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    a = (t - t0) / (t1 - t0).clamp_min(1e-12)
    return x0 + a * (x1 - x0)


def _segments(t: Tensor, x: Tensor, valid: Tensor, lo: float, hi: float) -> list[tuple[int, Tensor, Tensor, Tensor, Tensor]]:
    out = []
    for k in range(t.numel() - 1):
        if not bool(valid[k] and valid[k + 1]) or not bool(t[k + 1] > t[k]):
            continue
        left = torch.maximum(t[k], t.new_tensor(lo))
        right = torch.minimum(t[k + 1], t.new_tensor(hi))
        if bool(right <= left):
            continue
        out.append((k, left, right, _interp(t[k], t[k + 1], x[k], x[k + 1], left),
                    _interp(t[k], t[k + 1], x[k], x[k + 1], right)))
    return out


def _summarize(t: Tensor, x: Tensor, valid: Tensor, window: float) -> tuple[Tensor, Tensor, Tensor]:
    lo, hi = -float(window), 0.0
    segs = _segments(t, x, valid, lo, hi)
    visible = torch.nonzero(valid & (t >= lo) & (t <= hi), as_tuple=False).flatten()
    z = x.new_zeros(())
    if not segs:
        if visible.numel():
            last = x[visible[-1]]
            return torch.stack((last, last, last, z, z, z, z)), x.new_tensor(True, dtype=torch.bool), z
        return x.new_zeros(7), x.new_tensor(False, dtype=torch.bool), z
    duration = sum((b - a for _, a, b, _, _ in segs), z)
    integral = sum(((b - a) * (u + v) * 0.5 for _, a, b, u, v in segs), z)
    delta = sum((v - u for _, _, _, u, v in segs), z)
    variation = sum(((v - u).abs() for _, _, _, u, v in segs), z)
    feat = torch.stack((segs[0][3], segs[-1][4], integral / duration.clamp_min(1e-12),
                        delta, variation, duration / 5.0, duration / float(window)))
    return feat, x.new_tensor(True, dtype=torch.bool), duration / float(window)


def _pair_summary(t: Tensor, x: Tensor, y: Tensor, valid: Tensor, window: float) -> tuple[Tensor, Tensor, Tensor]:
    sx = _segments(t, x, valid, -float(window), 0.0)
    sy = _segments(t, y, valid, -float(window), 0.0)
    if not sx or len(sx) != len(sy):
        return x.new_zeros(14), x.new_tensor(False, dtype=torch.bool), x.new_zeros(())
    z = x.new_zeros(())
    duration = z
    ix = iy = im = area = dx_sum = dy_sum = tvx = tvy = z
    s1 = torch.zeros(2, device=x.device, dtype=x.dtype)
    s2 = torch.zeros(2, 2, device=x.device, dtype=x.dtype)
    previous_index = None
    for (index, a, b, x0, x1), (_, _, _, y0, y1) in zip(sx, sy):
        if previous_index is None or index != previous_index + 1:
            if previous_index is not None:
                area = area + .5 * (s2[0, 1] - s2[1, 0])
            s1.zero_(); s2.zero_()
        dt, dx, dy = b - a, x1 - x0, y1 - y0
        duration = duration + dt
        ix = ix + dt * (x0 + x1) * 0.5
        iy = iy + dt * (y0 + y1) * 0.5
        im = im + dt * (x0 * y0 + 0.5 * (x0 * dy + y0 * dx) + dx * dy / 3.0)
        d = torch.stack((dx, dy))
        s2 = s2 + torch.outer(s1, d) + .5 * torch.outer(d, d)
        s1 = s1 + d
        dx_sum, dy_sum = dx_sum + dx, dy_sum + dy
        tvx, tvy = tvx + dx.abs(), tvy + dy.abs(); previous_index = index
    area = area + .5 * (s2[0, 1] - s2[1, 0])
    cov = duration / float(window)
    feat = torch.stack((sx[0][3], sx[-1][4], ix / duration.clamp_min(1e-12), dx_sum, tvx,
                        sy[0][3], sy[-1][4], iy / duration.clamp_min(1e-12), dy_sum, tvy,
                        im / duration.clamp_min(1e-12), area / 0.05, duration / 5.0, cov))
    return feat, x.new_tensor(True, dtype=torch.bool), cov


class CoEVCoupledPathLift(nn.Module):
    windows = (1.0, 5.0)

    def __init__(self) -> None:
        super().__init__()
        self.pairs = tuple(combinations(range(15), 2))

    def forward(self, values: Tensor, valid: Tensor, actual_t: Tensor) -> dict[str, Tensor | list[str]]:
        if values.ndim != 3 or values.shape[-1] != 14 or valid.shape != values.shape:
            raise ValueError("values/valid must be [B,T,14]")
        if actual_t.shape != values.shape[:2]:
            raise ValueError("actual_t shape mismatch")
        bsz = values.shape[0]
        time_valid = valid.any(-1)
        time_coord = (actual_t / 5.0).unsqueeze(-1)
        all_values = torch.cat((values.float(), time_coord.float()), -1)
        all_valid = torch.cat((valid.bool(), time_valid.unsqueeze(-1)), -1)
        unary_rows, pair_rows, unary_ok, pair_ok, cover_rows = [], [], [], [], []
        for b in range(bsz):
            order = torch.argsort(actual_t[b])
            t = actual_t[b, order].float()
            xv, vv = all_values[b, order], all_valid[b, order]
            us, ps, uv, pv, uc, pc = [], [], [], [], [], []
            for window in self.windows:
                for i in range(14):
                    f, ok, cov = _summarize(t, xv[:, i], vv[:, i], window)
                    us.append(f); uv.append(ok); uc.append(cov)
                for i, j in self.pairs:
                    f, ok, cov = _pair_summary(t, xv[:, i], xv[:, j], vv[:, i] & vv[:, j], window)
                    ps.append(f); pv.append(ok); pc.append(cov)
            unary_rows.append(torch.stack(us)); pair_rows.append(torch.stack(ps))
            unary_ok.append(torch.stack(uv)); pair_ok.append(torch.stack(pv)); cover_rows.append(torch.stack(uc + pc))
        factor_ids = [f"w{int(w)}:u:{i}" for w in self.windows for i in range(14)] + [
            f"w{int(w)}:p:{i}:{j}" for w in self.windows for i, j in self.pairs]
        return {"unary": torch.stack(unary_rows), "pair": torch.stack(pair_rows),
                "unary_valid": torch.stack(unary_ok), "pair_valid": torch.stack(pair_ok),
                "factor_ids": factor_ids, "coverage": torch.stack(cover_rows)}
