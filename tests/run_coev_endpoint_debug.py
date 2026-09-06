import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--video",required=True);parser.add_argument("--target",required=True)
    parser.add_argument("--output-dir",required=True);parser.add_argument("--indices",type=int,nargs="*",default=[])
    parser.add_argument("--scan-best",action="store_true");args=parser.parse_args()
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    target_image=Image.open(args.target).convert("RGB");target_image.save(root/"target.jpg")
    target=np.asarray(target_image.resize((160,90)),dtype=np.float32)/255.0
    wanted=set(args.indices);scores=[]
    with av.open(args.video) as container:
        for index,frame in enumerate(container.decode(container.streams.video[0])):
            if index in wanted:frame.to_image().convert("RGB").save(root/f"frame_{index}.jpg")
            if args.scan_best:
                image=np.asarray(frame.to_image().convert("RGB").resize((160,90)),dtype=np.float32)/255.0
                scores.append((float(np.square(target-image).mean()),index,float(frame.time or 0.0)))
    if args.scan_best:
        best=sorted(scores)[:10]
        (root/"best_matches.json").write_text(json.dumps([
            {"mse_160x90":mse,"frame_index":index,"time_seconds":seconds}
            for mse,index,seconds in best],indent=2),encoding="utf-8")


if __name__=="__main__":main()
