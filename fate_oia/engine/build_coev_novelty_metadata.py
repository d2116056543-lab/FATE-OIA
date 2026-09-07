from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fate_oia.datasets.coev_video_dataset import decode_with_actual_pts, requested_times
from fate_oia.datasets.tida_clip_manifest import TIDAClipRecord, load_manifest


def _gray(image) -> np.ndarray:
    return np.asarray(image.convert("L").resize((56, 32)), dtype=np.float32) / 255.0


def _aligned_residual(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    fa = np.fft.rfft2(a - a.mean()); fb = np.fft.rfft2(b - b.mean())
    cross = fa * np.conj(fb); cross /= np.maximum(np.abs(cross), 1e-8)
    corr = np.fft.irfft2(cross, s=a.shape); y, x = np.unravel_index(np.argmax(corr), corr.shape)
    y = y if y <= a.shape[0] // 2 else y - a.shape[0]
    x = x if x <= a.shape[1] // 2 else x - a.shape[1]
    aligned = np.roll(b, (y, x), axis=(0, 1)); residual = np.abs(a - aligned)
    if y > 0: residual[:y] = 0
    elif y < 0: residual[y:] = 0
    if x > 0: residual[:, :x] = 0
    elif x < 0: residual[:, x:] = 0
    return residual, float(np.hypot(x / a.shape[1], y / a.shape[0]))


def _measure(record: TIDAClipRecord) -> dict:
    if not record.history_available:
        return {"file_name": record.file_name, "temporal_novelty_score": 0.0,
                "foreground_change": 0.0, "appearance_rate": 0.0,
                "residual_centroid_motion": 0.0, "camera_shift": 0.0,
                "history_available": False}
    try:
        frames, _, valid = decode_with_actual_pts(record.clip_path, requested_times(9), record.target_frame_index)
    except Exception as error:
        return {"file_name": record.file_name, "temporal_novelty_score": 0.0,
                "foreground_change": 0.0, "appearance_rate": 0.0,
                "residual_centroid_motion": 0.0, "camera_shift": 0.0,
                "history_available": True, "error": repr(error)}
    gray = [_gray(frame) for frame in frames]
    changes=[]; appearances=[]; centroids=[]; shifts=[]
    yy, xx = np.mgrid[0:32, 0:56]
    for index in range(8):
        if not bool(valid[index] and valid[index + 1]): continue
        residual, shift = _aligned_residual(gray[index], gray[index + 1])
        threshold = max(.06, float(np.quantile(residual, .80)))
        mask = residual >= threshold; mass = residual[mask]
        changes.append(float(mass.mean()) if mass.size else 0.0)
        appearances.append(float((residual > .12).mean())); shifts.append(shift)
        if mask.any(): centroids.append((float(xx[mask].mean()/55), float(yy[mask].mean()/31)))
    centroid_motion = float(np.mean([np.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(centroids,centroids[1:])])) if len(centroids)>1 else 0.0
    change=float(np.mean(changes)) if changes else 0.0; appearance=float(np.mean(appearances)) if appearances else 0.0
    score=.60*change+.25*appearance+.15*centroid_motion
    return {"file_name":record.file_name,"temporal_novelty_score":score,"foreground_change":change,
            "appearance_rate":appearance,"residual_centroid_motion":centroid_motion,
            "camera_shift":float(np.mean(shifts)) if shifts else 0.0,"history_available":True}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--manifest",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--workers",type=int,default=12);parser.add_argument("--max-records",type=int);args=parser.parse_args()
    records=[row for row in load_manifest(args.manifest) if row.partition in {"train_core","train_calib","train_audit"}]
    if args.max_records is not None: records=records[:args.max_records]
    with ProcessPoolExecutor(max_workers=args.workers) as pool: rows=list(pool.map(_measure,records,chunksize=8))
    rows.sort(key=lambda row:row["file_name"]); target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(target.suffix+".tmp");temp.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8");temp.replace(target)
    scores=np.asarray([row["temporal_novelty_score"] for row in rows])
    print(json.dumps({"rows":len(rows),"available":sum(row["history_available"] for row in rows),"errors":sum("error" in row for row in rows),
                      "score_quantiles":np.quantile(scores,[0,.2,.5,.8,1]).tolist(),"output":str(target)}))


if __name__=="__main__":main()
