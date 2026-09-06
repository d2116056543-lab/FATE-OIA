from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from fate_oia.datasets.coev_video_dataset import CoEVVideoDataset, coev_collate
from fate_oia.engine.train_coev_oia import build_model, load_config
from fate_oia.models.coev_observers import matched_expectation
from fate_oia.transforms import IMAGENET_MEAN, IMAGENET_STD
from fate_oia.utils.coev_preflight import source_identity


def as_image(value: torch.Tensor) -> Image.Image:
    value=(value.cpu()*IMAGENET_STD+IMAGENET_MEAN).clamp(0,1)
    array=(value.permute(1,2,0).numpy()*255).astype("uint8")
    return Image.fromarray(array)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/coev_oia_v1.yaml")
    parser.add_argument("--output-dir",required=True);parser.add_argument("--human-reviewed",action="store_true");args=parser.parse_args()
    cfg=load_config(args.config);root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    data=CoEVVideoDataset(cfg["data"]["manifest_path"],cfg["data"]["train_partitions"],False,max_samples=128)
    indices=torch.linspace(0,len(data)-1,16).round().long().tolist();model=build_model(cfg).cuda().eval();records=[]
    for case,index in enumerate(indices):
        inputs,_=coev_collate([data[index]])
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): output=model(inputs.to("cuda"))
        case_dir=root/f"case_{case:02d}";case_dir.mkdir(exist_ok=True)
        frames=[as_image(inputs.history_rgb[0,i]).resize((224,128)) for i in (0,4,8,13)]
        frames.append(as_image(inputs.target_rgb[0]).resize((224,128)))
        contact=Image.new("RGB",(224*5,128));
        for i,image in enumerate(frames):contact.paste(image,(224*i,0))
        contact.save(case_dir/"frames.jpg",quality=90)
        maps=output["predicate_maps"][0,-1].float().cpu();heat=Image.new("L",(28*8,16))
        for i,value in enumerate(maps):heat.paste(Image.fromarray((value.clamp(0,1).numpy()*255).astype("uint8")),(28*i,0))
        heat.resize((28*8*3,16*3),Image.Resampling.NEAREST).save(case_dir/"predicate_maps.png")
        curves=Image.new("RGB",(640,320),"white");draw=ImageDraw.Draw(curves);values=output["primitive_value"][0].float().cpu().clamp(-1,1)
        colors=[(int((i*71)%255),int((i*131)%255),int((i*191)%255)) for i in range(14)]
        for primitive in range(14):
            points=[(int(i*639/14),int((1-float(values[i,primitive]))*159.5)) for i in range(15)]
            draw.line(points,fill=colors[primitive],width=2)
        curves.save(case_dir/"primitive_curves.png")
        arrows=frames[-1].resize((448,256));draw=ImageDraw.Draw(arrows);expected,_,ok=matched_expectation(output["matcher_forward"][:, -1])
        for source,(target,good) in enumerate(zip(expected[0].float().cpu(),ok[0].cpu())):
            if source%4 or not bool(good):continue
            sy,sx=divmod(source,28);start=(int((sx+.5)*16),int((sy+.5)*16));end=(int((float(target[0])+1)*223.5),int((float(target[1])+1)*127.5))
            draw.line((start,end),fill="yellow",width=1)
        arrows.save(case_dir/"background_arrows.jpg",quality=90)
        records.append({"id":data.records[index].file_name,"index":index,"actual_t":inputs.actual_t[0].tolist(),"valid":inputs.valid[0].tolist(),
                        "finite":bool(output["logits"].isfinite().all()),"correspondence_quality_rate":float(output["correspondence_quality_rate"].mean()),
                        "artifacts":[str(case_dir/name) for name in ("frames.jpg","predicate_maps.png","primitive_curves.png","background_arrows.jpg")]})
    result={"pass":bool(args.human_reviewed and len(records)==16 and all(row["finite"] for row in records)),
            "machine_pass":len(records)==16 and all(row["finite"] for row in records),"human_reviewed":args.human_reviewed,
            "review_boundary":"Untrained maps may be weak; review checks coordinates, time, masks and rendering, not semantic quality.",
            "cases":records,"audit_identity":source_identity(args.config)}
    path=root/"visual_audit.json";path.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
    if not result["pass"]:raise SystemExit(2)


if __name__=="__main__":main()
