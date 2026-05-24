from pathlib import Path
from PIL import Image
import json, os, time

roots = [
    Path(r'E:/sbw/BDD100K/bdd100k_images/bdd100k/images/100k'),
    Path(r'E:/sbw/SNNA_repro/SNNA/dataset/BDD100k/images/100k'),
    Path(r'E:/sbw/SNNA_repro/SNNA/dataset/BDD100K/images/100k'),
]
expected = {'train': 70000, 'val': 10000, 'test': 20000}
print('=== BDD100K IMAGE ROOTS ===')
for root in roots:
    print(root, 'exists=', root.exists())

root = next((p for p in roots if p.exists()), None)
if root is None:
    raise SystemExit('NO_BDD100K_IMAGE_ROOT_FOUND')

print('=== COUNT CHECK ===')
all_names = []
for split, exp in expected.items():
    d = root / split
    files = sorted(d.glob('*.jpg')) if d.exists() else []
    all_names.extend([p.name for p in files])
    print(split, 'exists=', d.exists(), 'jpg_count=', len(files), 'expected=', exp, 'match=', len(files)==exp)
print('total_jpg=', len(all_names), 'expected_total=100000', 'match=', len(all_names)==100000)
print('unique_names=', len(set(all_names)), 'duplicate_names=', len(all_names)-len(set(all_names)))

print('=== SIZE CHECK ===')
for split in expected:
    files = list((root/split).glob('*.jpg'))
    zero = [p.name for p in files if p.stat().st_size == 0]
    tiny = [p.name for p in files if p.stat().st_size < 1024]
    print(split, 'zero_files=', len(zero), 'tiny_lt_1kb=', len(tiny), 'bytes_total=', sum(p.stat().st_size for p in files))

print('=== FULL JPEG VERIFY ===')
start = time.time()
bad = []
dims = {}
checked = 0
for split in expected:
    for p in (root/split).glob('*.jpg'):
        checked += 1
        try:
            with Image.open(p) as im:
                dims[im.size] = dims.get(im.size, 0) + 1
                im.verify()
        except Exception as e:
            bad.append((split, p.name, repr(e)))
            if len(bad) >= 20:
                break
    if len(bad) >= 20:
        break
print('checked=', checked, 'bad_count_first20=', len(bad), 'elapsed_sec=', round(time.time()-start, 1))
print('dims_top=', sorted(dims.items(), key=lambda kv: kv[1], reverse=True)[:10])
if bad:
    print('bad_examples=', bad[:20])

print('=== OTHER BDD100K PACKAGES UNDER E:/sbw/BDD100K ===')
base = Path(r'E:/sbw/BDD100K')
if base.exists():
    for p in sorted(base.iterdir()):
        if p.is_dir():
            n_files = 0
            try:
                for _,_,files in os.walk(p):
                    n_files += len(files)
            except Exception:
                pass
            print(p.name, 'files=', n_files)
