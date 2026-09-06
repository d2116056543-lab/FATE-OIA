from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageEnhance
import numpy as np
from torch.utils.data import Dataset

from fate_oia.datasets.coev_grounding_targets import CoEVGroundingTargetBuilder
from fate_oia.datasets.tida_clip_manifest import TIDAClipRecord, load_manifest
from fate_oia.transforms_video import SynchronizedVideoTransform
from fate_oia.utils.coev_contracts import CoEVInputs, CoEVTargets, flip_labels


def _configure_decode_stream(stream) -> None:
    # Two codec threads per loader worker improve H.264 decode throughput while
    # preserving presentation-order PTS and avoiding 56-core oversubscription.
    stream.thread_type = "AUTO"
    stream.codec_context.thread_count = 2


def requested_times() -> torch.Tensor:
    # Denser near the target while still covering the full five-second interval.
    u = torch.linspace(0, 1, 15)
    return -5.0 * (1.0 - u).square()


def probe_video_pts(path: str | Path) -> dict[str, float | int | bool]:
    import av
    pts = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        _configure_decode_stream(stream)
        for frame in container.decode(stream):
            if frame.pts is not None:
                pts.append(float(frame.pts * stream.time_base))
    if len(pts) < 2:
        raise RuntimeError(f"insufficient PTS frames: {path}")
    dt = torch.diff(torch.tensor(pts, dtype=torch.float64))
    median = float(dt.median())
    return {"decoded_frames":len(pts),"raw_dt_median":median,"raw_dt_min":float(dt.min()),"raw_dt_max":float(dt.max()),
            "cfr_pts":bool(((dt-median).abs() <= max(1e-6,abs(median)*.01)).float().mean() >= .99)}


def probe_endpoint_alignment(path: str | Path, target_path: str | Path, target_index: int) -> dict[str,float|int]:
    import av
    frames=[]
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; _configure_decode_stream(stream)
        for frame in container.decode(stream):frames.append(frame)
    target=np.asarray(Image.open(target_path).convert("RGB"),dtype=np.float32)/255.0
    candidates=sorted({index for center in (target_index,len(frames)-1) for index in range(center-2,center+3) if 0<=index<len(frames)})
    rows=[]
    for index in candidates:
        image=np.asarray(frames[index].to_image().convert("RGB").resize((target.shape[1],target.shape[0])),dtype=np.float32)/255.0
        rows.append((index,float(np.square(target-image).mean())))
    best=min(rows,key=lambda row:row[1])
    requested=next((mse for index,mse in rows if index==target_index),float("nan"))
    return {"manifest_target_index":target_index,"manifest_index_mse":requested,"best_nearby_index":best[0],"best_nearby_mse":best[1]}


def decode_with_actual_pts(path: str | Path, relative_times: torch.Tensor,
                           target_frame_index: int | None = None) -> tuple[list[Image.Image], torch.Tensor, torch.Tensor]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required for actual-PTS COEV decoding") from error
    frames, pts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        _configure_decode_stream(stream)
        for frame in container.decode(stream):
            if frame.pts is None: continue
            frames.append(frame); pts.append(float(frame.pts * stream.time_base))
    if not frames:
        raise RuntimeError(f"video has no decodable PTS frames: {path}")
    pts_t = torch.tensor(pts, dtype=torch.float64)
    anchor=len(pts)-1 if target_frame_index is None else int(target_frame_index)
    if not 0<=anchor<len(pts):raise RuntimeError(f"target frame index {anchor} outside decoded range {len(pts)}")
    target = pts_t[anchor]
    desired = target + relative_times.double()
    indices = torch.stack([(pts_t - x).abs().argmin() for x in desired])
    actual = (pts_t[indices] - target).float(); actual[-1] = 0
    valid = torch.ones(15, dtype=torch.bool)
    valid[1:] &= indices[1:] != indices[:-1]
    return [frames[int(i)].to_image().convert("RGB") for i in indices], actual, valid


class CoEVVideoDataset(Dataset):
    def __init__(self, manifest_path: str | Path, partitions: str | Sequence[str], training: bool,
                 grounding_root: str | Path | None = None, seed: int = 20260906,
                 max_samples: int | None = None) -> None:
        wanted = {partitions} if isinstance(partitions, str) else set(partitions)
        self.records = [r for r in load_manifest(manifest_path) if r.partition in wanted]
        if max_samples is not None: self.records = self.records[:max_samples]
        self.training = training; self.seed = seed; self.epoch = 0
        self.transform = SynchronizedVideoTransform(target_hw=(360, 640), context_hw=(256, 448))
        self.grounding = CoEVGroundingTargetBuilder(grounding_root) if training and grounding_root else None

    def set_epoch(self, epoch: int) -> None: self.epoch = int(epoch)
    def __len__(self) -> int: return len(self.records)

    def _seed(self, record: TIDAClipRecord, view: int = 0) -> int:
        raw = f"{self.seed}:{self.epoch}:{record.file_name}:{view}".encode()
        return int.from_bytes(sha256(raw).digest()[:8], "little")

    def __getitem__(self, index: int | tuple[int, int]) -> tuple[CoEVInputs, CoEVTargets]:
        sampler_seed = None
        if isinstance(index, tuple):
            index, sampler_seed = int(index[0]), int(index[1])
        record = self.records[index]
        target = Image.open(record.target_image_path).convert("RGB")
        if record.history_available:
            frames, actual_t, valid = decode_with_actual_pts(record.clip_path, requested_times(),record.target_frame_index)
            frames[-1] = target
        else:
            frames = [target.copy() for _ in range(15)]
            actual_t = requested_times(); actual_t[-1] = 0
            valid = torch.zeros(15, dtype=torch.bool); valid[-1] = True
        rng = torch.Generator().manual_seed(self._seed(record) if sampler_seed is None else sampler_seed)
        flip_value = float(torch.rand((), generator=rng))
        if self.training:
            brightness = .9 + .2 * float(torch.rand((), generator=rng))
            contrast = .9 + .2 * float(torch.rand((), generator=rng))
            frames = [ImageEnhance.Contrast(ImageEnhance.Brightness(x).enhance(brightness)).enhance(contrast) for x in frames]
        transformed = self.transform(frames, training=self.training, random_value=flip_value)
        action = torch.tensor(record.action, dtype=torch.float32); reason = torch.tensor(record.reason, dtype=torch.float32)
        if transformed["meta"]["flipped"]: action, reason = flip_labels(action, reason)
        if self.grounding is None:
            gv = torch.zeros(8, 16, 28); gk = torch.zeros_like(gv, dtype=torch.bool); gw = torch.zeros_like(gv)
        else:
            gv, gk, gw = self.grounding.build(record.file_name, transformed["meta"]["flipped"])
        inputs = CoEVInputs(transformed["target_image"], transformed["context_images"], actual_t, valid,
                            {"transform": transformed["meta"], "file_name": record.file_name})
        targets = CoEVTargets(action, reason, gv, gk, gw, [record.file_name])
        return inputs, targets


def coev_collate(rows: list[tuple[CoEVInputs, CoEVTargets]]) -> tuple[CoEVInputs, CoEVTargets]:
    ins, tgts = zip(*rows)
    inputs = CoEVInputs(torch.stack([x.target_rgb for x in ins]), torch.stack([x.history_rgb for x in ins]),
                        torch.stack([x.actual_t for x in ins]), torch.stack([x.valid for x in ins]),
                        {"samples": [x.geometry_meta for x in ins]})
    targets = CoEVTargets(torch.stack([x.action for x in tgts]), torch.stack([x.reason for x in tgts]),
                          torch.stack([x.grounding_value for x in tgts]), torch.stack([x.grounding_known for x in tgts]),
                          torch.stack([x.grounding_weight for x in tgts]), [x.ids[0] for x in tgts])
    return inputs, targets
