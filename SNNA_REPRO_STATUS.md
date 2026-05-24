# SNNA Reproduction Status

This folder is a direct reproduction workspace for:

- Paper: `Noise-Free Explanation for Driving Action Prediction`
- Code: `https://github.com/Hongbo-Z/SNNA`
- Local clone: `E:\sbw\SNNA_repro\SNNA`

## Official Protocol Captured

The GitHub README defines a two-stage pipeline:

1. Self-supervised DINO training on BDD100K images:

```bash
python -m torch.distributed.launch --nproc_per_node=1 main_dino.py \
  --patch_size 8 \
  --batch_size_per_gpu 16 \
  --epochs 200 \
  --saveckp_freq 10 \
  --data_path dataset/BDD100k/images/100k \
  --output_dir ckp/dino_bdd100k
```

2. Supervised BDD-OIA multi-label classifier training:

```bash
python multi_label_train.py \
  --num_labels 4 \
  --patch_size 8 \
  --batch_size_per_gpu 4 \
  --epochs 100 \
  --pretrained_weights ckp/backbone_200.pth \
  --data_path dataset/BDD-OIA \
  --output_dir ckp/classifier_bdd_oia
```

The paper appendix states ViT-S/8, 200 epochs DINO on BDD100K train, then frozen-backbone classifier fine-tuning on BDD-OIA for 100 epochs. During supervised fine-tuning the input is resized to `360 x 640` by the repository code.

## What Has Been Prepared

- Repository cloned at `E:\sbw\SNNA_repro\SNNA`.
- Reproduction helper scripts are placed in `E:\sbw\SNNA_repro\SNNA\scripts_repro`.
- BDD-OIA conversion script prepares the README layout from the official flat image folder and COCO-style action JSON files.
- BDD100K path is linked into `dataset\BDD100k\images\100k` when extraction is complete.
- Public DINO ViT-S/8 reference weights are downloaded into `ckp\reference`.
- `main_dino.py.upstream_snna` preserves the original GitHub file. The active
  `main_dino.py` has one reproducibility-only patch: if querying
  `torch.hub.list("facebookresearch/xcit:main")` fails due GitHub API/rate limit,
  it falls back to local ViT/torchvision choices. SNNA's official command uses
  `vit_small`, so this does not alter the ViT-S/8 training path.

## Important Checkpoint Clarification

The SNNA repository ships only:

```text
ckp/classifier.pth.tar
```

It does not ship the BDD100K DINO checkpoint expected by the README:

```text
ckp/backbone_200.pth
```

That file must be produced by running `main_dino.py` for 200 epochs on BDD100K. After the DINO run finishes, run:

```bash
python scripts_repro/export_backbone_200.py \
  --dino_checkpoint ckp/dino_bdd100k/checkpoint.pth \
  --output_path ckp/backbone_200.pth \
  --force
```

The classifier training can then be run exactly with `ckp/backbone_200.pth`.

## Data Layout Expected

```text
E:\sbw\SNNA_repro\SNNA\dataset
  BDD100k
    images
      100k
        train
        val
        test
  BDD-OIA
    train
    val
    test
    train.json
    val.json
    test.json
```

BDD-OIA source currently detected:

```text
E:\sbw\BDD-OIA
  data
  train_25k_images_actions.json
  val_25k_images_actions.json
  test_25k_images_actions.json
```

BDD-OIA action labels are converted to the first four labels:

```text
forward, stop, left, right
```

The fifth public BDD-OIA label `confuse` is ignored because SNNA uses `--num_labels 4`.

Prepared BDD-OIA counts:

```text
train: 16082 images / 16082 labels
val:    2270 images / 2270 labels
test:   4572 images / 4572 labels
```

Prepared BDD100K counts through the linked source directory:

```text
train: 70000 jpg
val:   10000 jpg
test:  20000 jpg
```

## Selected BDD100K Folder

The correct BDD100K image folder for SNNA is:

```text
E:\sbw\BDD100K\bdd100k_images\bdd100k\images\100k
```

It is linked into the repo as:

```text
E:\sbw\SNNA_repro\SNNA\dataset\BDD100k\images\100k
```

Other BDD100K folders are not used by SNNA's code path:

```text
bdd100k_labels       not used by SNNA training
bdd100k_seg          not used by SNNA training
bdd100k_drivable_maps not used by SNNA training
bdd100k_info         not used by SNNA training
```

Important detail: `main_dino.py` uses `torchvision.datasets.ImageFolder`.
Therefore the README command points to `dataset/BDD100k/images/100k`, where
`train`, `val`, and `test` become pseudo-class folders. DINO ignores those class
labels, so this is compatible with the official repo command. If using only the
flat `train` directory, `ImageFolder` would not work without adding another
wrapper folder or changing the code.

Runtime loader verification in the exact `SNNA` environment:

```text
bdd100k_imagefolder_len: 100000
bdd100k_class_to_idx: {'test': 0, 'train': 1, 'val': 2}
bdd100k/train: 70000 jpg
bdd100k/val:   10000 jpg
bdd100k/test:  20000 jpg

bdd_oia/train: 16082 images / 16082 labels
bdd_oia/val:    2270 images / 2270 labels
bdd_oia/test:   4572 images / 4572 labels
```

Verification log:

```text
E:\sbw\SNNA_repro\repro_logs\snna_verify_dataset_paths.log
```

## Scripts

Run these from WSL:

```bash
cd /mnt/e/sbw/SNNA_repro/SNNA
bash scripts_repro/create_snna_env.sh /mnt/e/sbw/SNNA_repro/SNNA SNNA
bash scripts_repro/train_dino_bdd100k_single_gpu.sh /mnt/e/sbw/SNNA_repro/SNNA SNNA
python scripts_repro/export_backbone_200.py --dino_checkpoint ckp/dino_bdd100k/checkpoint.pth --output_path ckp/backbone_200.pth --force
bash scripts_repro/train_classifier_bdd_oia_single_gpu.sh /mnt/e/sbw/SNNA_repro/SNNA SNNA
```

## Remaining Blockers

- Exact `SNNA.yml` conda environment creation was attempted first against the
  default channels. Both default-channel attempts reached package download, but
  the remote Anaconda connection broke with `IncompleteRead` while fetching
  large packages such as PyTorch/CUDA. The same package versions were then
  installed successfully from TUNA mirrors using
  `scripts_repro/create_snna_env_tuna.sh`.

Environment verification:

```text
conda env: SNNA
python: 3.6.13
torch: 1.7.1
torchvision: 0.8.2
cuda_available: True
gpu: NVIDIA GeForce RTX 4090
```

Logs:

```text
E:\sbw\SNNA_repro\repro_logs\snna_env_create.log
E:\sbw\SNNA_repro\repro_logs\snna_env_create_retry.log
E:\sbw\SNNA_repro\repro_logs\snna_env_create_tuna.log
```

- `ckp/backbone_200.pth` cannot be downloaded from the SNNA README; it must be generated by the 200-epoch BDD100K DINO stage unless the authors provide a checkpoint separately.

## Reproduction Completeness Review

Path, data placement, environment, public reference weights, and scripts are now
in place. The setup matches the public GitHub README layout and commands. The
only known deviations/constraints are:

1. The active `main_dino.py` has an offline-safe XCiT list fallback because the
   original file queries GitHub at import/help time. This does not change the
   official ViT-S/8 path used by SNNA.
2. The paper says DINO is trained on BDD100K train, while the README command
   passes the `100k` root. I preserved the README/code-exact path because the
   code uses `ImageFolder` and expects subfolders under the data root.
3. Full result reproduction still requires running the two official training
   stages: 200 epochs BDD100K DINO and 100 epochs BDD-OIA classifier.

## Verified Smoke Checks

```text
E:\sbw\SNNA_repro\repro_logs\snna_dataset_smoke.log
E:\sbw\SNNA_repro\repro_logs\snna_code_import_smoke_after_patch.log
E:\sbw\SNNA_repro\repro_logs\snna_dataset_smoke_SNNA_env.log
E:\sbw\SNNA_repro\repro_logs\snna_code_import_smoke_SNNA_env.log
E:\sbw\SNNA_repro\repro_logs\snna_setup_verify_final.json
```

The dataset smoke check confirms BDD-OIA labels/images match. The code import
smoke confirms `main_dino.py --help` and `multi_label_train.py --help` run after
the XCiT offline fallback patch.
