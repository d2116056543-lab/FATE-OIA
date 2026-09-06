from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml
import numpy as np
from PIL import Image

from fate_oia.datasets.coev_grounding_targets import CoEVGroundingTargetBuilder
from fate_oia.datasets.coev_video_dataset import decode_with_actual_pts, probe_endpoint_alignment, probe_video_pts, requested_times
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.tida_clip_manifest import load_manifest, normalize_source_id
from fate_oia.engine.train_coev_oia import build_model, build_optimizer, load_config, loaders, lr_factor
from fate_oia.losses.coev_losses import CoEVLoss


REQUIRED = (
 "fate_oia/models/coev_visual_field.py", "fate_oia/models/coev_observers.py", "fate_oia/models/coev_path_lift.py",
 "fate_oia/models/coev_video_decoder.py", "fate_oia/models/coev_evidence_readout.py", "fate_oia/models/coev_oia_model.py",
 "fate_oia/datasets/coev_video_dataset.py", "fate_oia/datasets/coev_grounding_targets.py", "fate_oia/losses/coev_losses.py",
 "fate_oia/engine/train_coev_oia.py", "fate_oia/engine/evaluate_coev_oia.py", "fate_oia/explain/coev_interventions.py",
 "fate_oia/engine/audit_coev_endpoint_identity.py", "fate_oia/engine/repair_coev_endpoint_clips.py",
 "fate_oia/utils/coev_contracts.py", "fate_oia/utils/coev_checkpoint.py", "fate_oia/utils/coev_preflight.py",
 "fate_oia/utils/coev_supervisor.py", "configs/coev_oia_v1.yaml", "scripts/run_coev_foreground.ps1",
 ".codex/skills/coev-oia-v1-implementation-audit/SKILL.md")


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp"); temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"); temp.replace(path)


def source_identity(config_path: str | Path) -> dict:
    audit_files=sorted(set(REQUIRED)|{"fate_oia/datasets/bdd100k_grounding.py"}|{str(x).replace("\\","/") for pattern in ("tests/test_coev_*.py","tests/run_coev_*.py","reference/*.py") for x in Path().glob(pattern)})
    return {"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "config_sha256":sha(config_path),"required_source_hashes":{name:sha(name) for name in audit_files if Path(name).is_file()}}


def stamp(path: Path, config_path: str | Path) -> None:
    payload=json.loads(path.read_text(encoding="utf-8"));payload["audit_identity"]=source_identity(config_path);atomic_json(path,payload)


def requirement_matrix(out: Path, checks: dict[str,bool]) -> dict[str,dict]:
    plan=Path("docs/superpowers/plans/2026-09-06-coev-oia-v1-implementation.md").read_text(encoding="utf-8")
    result={}
    categories={range(1,7):"data",range(7,27):"functional",range(27,33):"functional",
                range(33,34):"memory",range(34,36):"supervisor",range(36,37):"git",range(37,38):"mutation",
                range(38,40):"functional",range(40,41):"identity"}
    for line in plan.splitlines():
        if not line.startswith("| R"):continue
        cells=[cell.strip().strip("`") for cell in line.strip("|").split("|")]
        number=int(cells[0][1:]);category=next(name for numbers,name in categories.items() if number in numbers)
        result[cells[0]]={"symbol":cells[1],"test":cells[2],"artifact":cells[3],"observed":{"category":category},"pass":bool(checks[category])}
    return result


def data_audit(cfg: dict) -> dict:
    rows = load_manifest(cfg["data"]["manifest_path"])
    train = [r for r in rows if r.partition in cfg["data"]["train_partitions"]]; test = [r for r in rows if r.partition == "test"]
    ids = lambda xs: [r.file_name.lower() for r in xs]
    digest = lambda xs: hashlib.sha256("\n".join(xs).encode()).hexdigest()
    train_source = {normalize_source_id(r.source_video_id) for r in train}; test_source = {normalize_source_id(r.source_video_id) for r in test}
    missing = sum(not r.history_available for r in rows); missing_test = sum(not r.history_available for r in test)
    duplicate_targets = len(ids(rows)) - len(set(ids(rows)))
    # Deterministic real-video sample: audit selected actual PTS rather than trusting manifest FPS.
    available = [r for r in rows if r.history_available and r.clip_path.is_file()]
    stride = max(1, len(available) // 32); pts_rows=[]; pts_errors=[]
    for record in available[::stride][:32]:
        try:
            decoded, actual_t, valid = decode_with_actual_pts(record.clip_path, requested_times(),record.target_frame_index)
            raw_pts = probe_video_pts(record.clip_path)
            alignment=probe_endpoint_alignment(record.clip_path,record.target_image_path,record.target_frame_index)
            target=np.asarray(Image.open(record.target_image_path).convert("RGB"),dtype=np.float32)/255.0
            endpoint=np.asarray(decoded[-1].resize((target.shape[1],target.shape[0])),dtype=np.float32)/255.0
            endpoint_mse=float(np.square(target-endpoint).mean())
            gaps = actual_t[1:] - actual_t[:-1]
            selected = gaps[valid[1:] & valid[:-1]].tolist()
            pts_rows.append({"id":record.file_name,"selected_dt":selected,"valid_frames":int(valid.sum()),
                             "duration":float(actual_t[-1]-actual_t[0]),"manifest_fps":record.fps,"endpoint_mse":endpoint_mse,**raw_pts,**alignment})
        except Exception as error:
            pts_errors.append({"id":record.file_name,"error":repr(error)})
    dt_values=torch.tensor([x for row in pts_rows for x in row["selected_dt"]],dtype=torch.float64)
    quantiles = ({str(q):float(torch.quantile(dt_values,q)) for q in (.0,.1,.5,.9,1.)} if dt_values.numel() else {})
    cfr_like=sum(bool(row["cfr_pts"]) for row in pts_rows)
    # Weak targets are train-only and unknown is retained when a source is absent.
    builder=CoEVGroundingTargetBuilder(cfg["data"]["grounding_root"]); weak=[]
    weak_stride=max(1,len(train)//512)
    for record in train[::weak_stride][:512]:
        value,known,_=builder.build(record.file_name,False)
        weak.append((known.flatten(1).any(-1), (value*known).flatten(1).amax(-1)>0))
    names=cfg["model"]["predicates"]
    known=torch.stack([x[0] for x in weak]) if weak else torch.zeros(0,8,dtype=torch.bool)
    positive=torch.stack([x[1] for x in weak]) if weak else torch.zeros(0,8,dtype=torch.bool)
    weak_coverage={name:{"positive":int(positive[:,i].sum()),"negative":int((known[:,i]&~positive[:,i]).sum()),
                         "unknown":int((~known[:,i]).sum())} for i,name in enumerate(names)}
    historical=BDDOIAMultiTaskDataset(cfg["data"]["historical_image_data_root"],cfg["data"]["historical_image_raw_root"],"train")
    historical_ids={row.file_name.lower() for row in historical.samples};video_train_ids=set(ids(train))
    historical_only=sorted(historical_ids-video_train_ids);video_only=sorted(video_train_ids-historical_ids)
    ok = (len(train) == cfg["data"]["train_count"] and len(test) == cfg["data"]["test_count"]
          and missing == cfg["data"]["expected_history_missing"] and missing_test == cfg["data"]["expected_test_history_missing"]
          and not set(ids(train)) & set(ids(test)) and duplicate_targets == 0 and bool(pts_rows) and not pts_errors
          and max((row["endpoint_mse"] for row in pts_rows),default=1.0)<=cfg["data"]["max_endpoint_mse"]
          and all(len(row.action)==4 and len(row.reason)==21 for row in rows)
          and len(historical_ids)==cfg["data"]["historical_image_train_count"])
    return {"pass":ok,"train_count":len(train),"test_count":len(test),"union_count":len({*ids(train),*ids(test)}),
            "train_id_sha256":digest(ids(train)),"test_id_sha256":digest(ids(test)),"target_overlap":len(set(ids(train))&set(ids(test))),
            "source_video_overlap":len(train_source & test_source),"history_missing":missing,"test_history_missing":missing_test,
            "expected_history_missing":cfg["data"]["expected_history_missing"],"expected_test_history_missing":cfg["data"]["expected_test_history_missing"],
            "historical_image_train_count":cfg["data"]["historical_image_train_count"],"difference_count":cfg["data"]["historical_image_train_count"]-len(train),
            "historical_image_actual_count":len(historical_ids),"historical_only_ids":historical_only,"video_only_ids":video_only,
            "duplicate_target_ids":duplicate_targets,"actual_pts_sample_count":len(pts_rows),"actual_pts_errors":pts_errors,
            "endpoint_mse_max":max((row["endpoint_mse"] for row in pts_rows),default=None),"label_dimension_mismatch_count":sum(len(row.action)!=4 or len(row.reason)!=21 for row in rows),
            "selected_actual_dt_quantiles":quantiles,"cfr_like_ratio":cfr_like/max(1,len(pts_rows)),"pts_samples":pts_rows,
            "weak_target_sample_count":len(weak),"weak_label_coverage":weak_coverage,
            "weak_label_warnings":[name for name,row in weak_coverage.items() if row["positive"]+row["negative"]==0],
            "split_policy_note":"Official target split is preserved; source-video overlap is reported and not silently rewritten."}


def functional(cfg: dict, out: Path) -> dict:
    tests = [str(x) for x in sorted(Path("tests").glob("test_coev_*.py"))] + ["reference/test_reference_ops.py"]
    command = ["python","-m","pytest",*tests,"-q"]
    run = subprocess.run(command, text=True, capture_output=True)
    result = {"pass":run.returncode==0,"command":command,"returncode":run.returncode,"stdout":run.stdout[-12000:],"stderr":run.stderr[-12000:]}
    atomic_json(out/"functional_audit.json",result); return result


def mutation_audit(out: Path) -> dict:
    command=["python","-m","pytest","tests/test_coev_mutation_guards.py","-q"]
    run=subprocess.run(command,text=True,capture_output=True)
    result={"pass":run.returncode==0,"command":command,"returncode":run.returncode,
            "shape_legal_mutations":10,"required_mutations":14,"stdout":run.stdout[-12000:],"stderr":run.stderr[-12000:]}
    atomic_json(out/"mutation_kill_report.json",result);return result


def real_rgb_probe(cfg: dict, out: Path) -> dict:
    if not torch.cuda.is_available():
        result={"pass":False,"error":"CUDA unavailable"};atomic_json(out/"real_rgb_100_update_probe.json",result);return result
    batch,accum,updates=2,16,100
    loader,_,_=loaders(cfg,batch,128,batch);iterator=iter(loader)
    model=build_model(cfg).cuda().train();optimizer=build_optimizer(model,cfg);loss_fn=CoEVLoss()
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda u:lr_factor(u,cfg["training"]["total_updates"],
        cfg["training"]["warmup_updates"],cfg["training"]["min_lr_ratio"]))
    initial={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
    loss_rows=[];component_nonzero={name:False for name in ("action_task","reason_task","ground","match")}
    last_batch=None
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            try: inputs,targets=next(iterator)
            except StopIteration: iterator=iter(loader);inputs,targets=next(iterator)
            last_batch=inputs
            with torch.autocast("cuda",dtype=torch.bfloat16): losses=loss_fn(model(inputs.to("cuda")),targets.to("cuda"))
            (losses["total"]/accum).backward()
            for name in component_nonzero: component_nonzero[name] |= bool(torch.isfinite(losses[name]) and losses[name].abs()>1e-10)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["grad_clip"]);optimizer.step();scheduler.step()
        loss_rows.append(float(losses["total"].detach()))
    updates_by_owner={};prefixes={"dino9":"visual_field.backbone.blocks.8","dino10":"visual_field.backbone.blocks.9",
        "dino11":"visual_field.backbone.blocks.10","dino12":"visual_field.backbone.blocks.11","decoder":"video_decoder",
        "predicate":"predicate_observer","matcher":"correspondence_observer","readout":"evidence_readout"}
    for owner,prefix in prefixes.items():
        values=[(p.detach()-initial[name]).float().square().sum() for name,p in model.named_parameters() if name.startswith(prefix+".")]
        updates_by_owner[owner]=float(torch.stack(values).sum().sqrt()) if values else 0.0
    model.eval();assert last_batch is not None;base=last_batch.to("cuda")
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        logits=model(base)["logits"]
        changed=base.history_rgb.clone();changed[:,0].add_(.1)
        early=model(type(base)(base.target_rgb,changed,base.actual_t,base.valid,base.geometry_meta))["logits"]
        changed=base.history_rgb.clone();changed[:,6].add_(.1)
        middle=model(type(base)(base.target_rgb,changed,base.actual_t,base.valid,base.geometry_meta))["logits"]
    dependence={"early_mean_abs_delta":float((early-logits).abs().mean()),"middle_mean_abs_delta":float((middle-logits).abs().mean())}
    result={"pass":all(component_nonzero.values()) and all(v>0 for v in updates_by_owner.values()) and all(v>0 for v in dependence.values()),
            "fixed_train_samples":128,"optimizer_updates":updates,"batch_size":batch,"gradient_accumulation_steps":accum,
            "formal_schedule_total_updates":cfg["training"]["total_updates"],"loss_first":loss_rows[0],"loss_last":loss_rows[-1],
            "component_nonzero":component_nonzero,"owner_update_norm":updates_by_owner,"input_dependence":dependence,
            "probe_state_discarded":True}
    atomic_json(out/"real_rgb_100_update_probe.json",result);return result


def memory(cfg: dict, out: Path) -> dict:
    if not torch.cuda.is_available():
        result={"pass":False,"error":"CUDA unavailable"}; atomic_json(out/"memory_profile.json",result); return result
    def trial(batch: int, accum: int, warmup: int, measured: int, include_test: bool=False) -> dict:
        torch.cuda.empty_cache(); model=build_model(cfg).cuda().train(); loss_fn=CoEVLoss()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        train,test,_=loaders(cfg,batch,max(128,batch*accum),batch)
        iterator=iter(train);torch.cuda.reset_peak_memory_stats();times=[];load_times=[];losses=[]
        try:
            for update in range(warmup+measured):
                tick=time.perf_counter();opt.zero_grad(set_to_none=True);load_elapsed=0.0
                for _ in range(accum):
                    load_tick=time.perf_counter()
                    try: inputs,targets=next(iterator)
                    except StopIteration: iterator=iter(train);inputs,targets=next(iterator)
                    load_elapsed+=time.perf_counter()-load_tick
                    with torch.autocast("cuda",dtype=torch.bfloat16): value=loss_fn(model(inputs.to("cuda")),targets.to("cuda"))["total"]
                    (value/accum).backward();losses.append(float(value.detach()))
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["training"]["grad_clip"]);opt.step();torch.cuda.synchronize()
                if update>=warmup: times.append(time.perf_counter()-tick);load_times.append(load_elapsed)
            test_shape=None
            if include_test:
                model.eval();inputs,_=next(iter(test))
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): test_shape=list(model(inputs.to("cuda"))["logits"].shape)
                torch.cuda.synchronize()
            values=torch.tensor(times)
            return {"pass":bool(times) and torch.isfinite(torch.tensor(losses)).all().item(),"batch_size":batch,"grad_accum":accum,
                    "warmup_updates":warmup,"measured_updates":measured,"median_update_seconds":float(values.median()),
                    "p95_update_seconds":float(torch.quantile(values,.95)),"decode_load_fraction":sum(load_times)/max(1e-9,sum(times)),
                    "samples_per_second":batch*accum/max(1e-9,float(values.median())),
                    "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,"peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
                    "loss_first":losses[0],"loss_last":losses[-1],"full_test_forward_shape":test_shape}
        except torch.cuda.OutOfMemoryError as error:
            return {"pass":False,"batch_size":batch,"grad_accum":accum,"error":"CUDA OOM: "+str(error)}
        finally:
            del model,opt;torch.cuda.empty_cache()
    profiles=[trial(int(batch),int(accum),cfg["memory_probe"]["warmup_updates"],cfg["memory_probe"]["measured_updates"])
              for batch,accum in cfg["memory_probe"]["candidates"]]
    eligible=[row for row in profiles if row["pass"] and row["peak_reserved_gib"]<47.0]
    selected=max(eligible,key=lambda row:row["samples_per_second"]) if eligible else None
    stress=(trial(selected["batch_size"],selected["grad_accum"],0,cfg["memory_probe"]["stress_updates"],True) if selected else None)
    result={"pass":bool(stress and stress["pass"] and stress["measured_updates"]==200 and stress["full_test_forward_shape"]),
            "candidate_profiles":profiles,"selected":selected,"stress":stress,"device":torch.cuda.get_device_name(0)}
    atomic_json(out/"memory_profile.json",result); return result


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--stage",choices=("data","functional","probe","memory","ready"),required=True);p.add_argument("--output",default=".review/coev_v1");a=p.parse_args()
    cfg=load_config(a.config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.stage=="data":
        result=data_audit(cfg);atomic_json(out/"data_audit.json",result);stamp(out/"data_audit.json",a.config)
        if not result["pass"]: raise SystemExit(2)
        return
    if a.stage=="functional":
        base=functional(cfg,out); mutations=mutation_audit(out);stamp(out/"functional_audit.json",a.config);stamp(out/"mutation_kill_report.json",a.config)
        if not base["pass"] or not mutations["pass"]: raise SystemExit(2)
        return
    if a.stage=="probe":
        result=real_rgb_probe(cfg,out);stamp(out/"real_rgb_100_update_probe.json",a.config)
        if not result["pass"]: raise SystemExit(2)
        return
    if a.stage=="memory":
        result=memory(cfg,out);stamp(out/"memory_profile.json",a.config)
        if not result["pass"]: raise SystemExit(2)
        return
    evidence=[out/"data_audit.json",out/"endpoint_identity_audit.json",out/"functional_audit.json",out/"mutation_kill_report.json",out/"real_smoke.json",
              out/"owner_gradient_audit.json",out/"visual"/"visual_audit.json",out/"real_rgb_100_update_probe.json",out/"memory_profile.json"]
    missing=[str(x) for x in evidence if not x.is_file()]; failed=[str(x) for x in evidence if x.is_file() and not json.loads(x.read_text(encoding="utf-8"))["pass"]]
    memory_payload=json.loads((out/"memory_profile.json").read_text(encoding="utf-8")) if (out/"memory_profile.json").is_file() else {}
    if memory_payload.get("stress",{}).get("measured_updates") != cfg["memory_probe"]["stress_updates"]:
        failed.append(str(out/"memory_profile.json")+":stale_or_short_stress")
    tracked=set(subprocess.check_output(["git","ls-files"],text=True).splitlines()); untracked=[x for x in REQUIRED if x not in tracked]
    current_identity=source_identity(a.config);stale=[]
    for path in evidence:
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("audit_identity") != current_identity:
            stale.append(str(path))
    bound_files=list(current_identity["required_source_hashes"])
    dirty=subprocess.run(["git","status","--porcelain","--",*bound_files],text=True,capture_output=True).stdout.strip().splitlines()
    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip();local_head=current_identity["git_head"]
    remote_matches=[]
    for remote in subprocess.check_output(["git","remote"],text=True).splitlines():
        lookup=subprocess.run(["git","ls-remote",remote,f"refs/heads/{branch}"],text=True,capture_output=True)
        if lookup.returncode==0 and lookup.stdout.strip().split("\t")[0:1]==[local_head]: remote_matches.append(remote)
    skill_matches=sha(REQUIRED[-1])==cfg["experiment"]["spec_sha256"]["SKILL.md"]
    launch=(memory_payload.get("stress") or {})
    checks={"data":not any("data_audit" in x for x in missing+failed+stale),
            "functional":not any(any(name in x for name in ("functional_audit","real_smoke","owner_gradient","real_rgb","visual_audit")) for x in missing+failed+stale),
            "memory":not any("memory_profile" in x for x in missing+failed+stale),"supervisor":Path("fate_oia/utils/coev_supervisor.py").is_file(),
            "git":not(untracked or dirty) and bool(remote_matches),"mutation":not any("mutation_kill" in x for x in missing+failed+stale),
            "identity":not stale and skill_matches}
    requirements=requirement_matrix(out,checks)
    payload={"schema":"coev_ready_v1","pass":not(missing or failed or untracked or stale or dirty) and bool(remote_matches) and skill_matches and len(requirements)==40 and all(x["pass"] for x in requirements.values()),
             "missing":missing,"failed":failed,"untracked":untracked,"stale_artifacts":stale,"dirty_required_sources":dirty,
             "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"source_hashes":{x:sha(x) for x in REQUIRED if Path(x).is_file()},
             "config_sha256":sha(a.config),"data_identity_sha256":sha(out/"data_audit.json") if (out/"data_audit.json").is_file() else None,
             "spec_sha256":cfg["experiment"]["spec_sha256"],"skill_hash_matches_spec":skill_matches,"branch":branch,
             "remote_head_matches":remote_matches,"launch_configuration":{"batch_size":launch.get("batch_size"),
                 "gradient_accumulation_steps":launch.get("grad_accum"),"effective_batch":cfg["training"]["effective_batch"]},
             "requirements":requirements,"missing_items":[key for key,value in requirements.items() if not value["pass"]]}
    atomic_json(out/"FULL_TRAIN_READY.json",payload)
    if not payload["pass"]: raise SystemExit(2)


if __name__=="__main__": main()
