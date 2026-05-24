from pathlib import Path
import re, hashlib, random
from collections import Counter

repo = Path(r'E:/sbw/SNNA_repro/SNNA')
oia_root = repo/'dataset/BDD-OIA'
bdd_root_candidates = [
    Path(r'E:/sbw/BDD100K/bdd100k_images/bdd100k/images/100k'),
    repo/'dataset/BDD100k/images/100k',
    repo/'dataset/BDD100K/images/100k',
]
bdd_root = next((p for p in bdd_root_candidates if p.exists()), None)
print('bdd_root', bdd_root)
print('oia_root', oia_root, oia_root.exists())

bdd = {}
bdd_by_split = {}
if bdd_root:
    for split in ['train','val','test']:
        files = list((bdd_root/split).glob('*.jpg')) if (bdd_root/split).exists() else []
        bdd_by_split[split] = {p.name: p for p in files}
        for p in files:
            bdd[p.name] = p
        print('bdd', split, len(files))

def strip_suffix(name):
    stem = Path(name).stem
    m = re.match(r'^(.*)_([0-9]+)$', stem)
    if m:
        return m.group(1) + Path(name).suffix, int(m.group(2))
    return name, None

def md5(p):
    h = hashlib.md5()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

all_oia=[]
for split in ['train','val','test']:
    img_dir=oia_root/split
    files=list(img_dir.glob('*.jpg')) if img_dir.exists() else []
    exact=sum(1 for p in files if p.name in bdd)
    stripped=sum(1 for p in files if strip_suffix(p.name)[0] in bdd)
    suffixes=Counter(strip_suffix(p.name)[1] for p in files)
    print('oia', split, len(files), 'exact', exact, 'strip_base_match', stripped, 'suffix_top', suffixes.most_common(12))
    for p in files:
        base, suf=strip_suffix(p.name)
        all_oia.append((split,p,base,suf,bdd.get(p.name),bdd.get(base)))

print('oia_total', len(all_oia))
print('exact_total', sum(1 for _,p,_,_,exact,basep in all_oia if exact))
print('strip_total', sum(1 for _,p,base,_,exact,basep in all_oia if basep))
print('unmatched_after_strip_examples', [p.name for _,p,base,_,exact,basep in all_oia if not basep][:20])

exact_pairs=[(p,exact) for _,p,_,_,exact,_ in all_oia if exact]
base_pairs=[(p,basep) for _,p,_,suf,_,basep in all_oia if suf is not None and basep]
print('exact_pairs', len(exact_pairs), 'suffix_base_pairs', len(base_pairs))
for label,pairs in [('exact_first200', exact_pairs[:200]), ('suffix_base_sample200', random.sample(base_pairs, min(200,len(base_pairs))) if base_pairs else [])]:
    same_size=0; same_md5=0; size_examples=[]
    for a,b in pairs:
        if a.stat().st_size == b.stat().st_size:
            same_size += 1
        else:
            if len(size_examples)<5:
                size_examples.append((a.name,a.stat().st_size,b.name,b.stat().st_size))
        if a.stat().st_size == b.stat().st_size and md5(a)==md5(b):
            same_md5 += 1
    print(label, 'sample_n', len(pairs), 'same_size', same_size, 'same_md5', same_md5, 'size_examples', size_examples[:5])

splitmap={name:split for split,d in bdd_by_split.items() for name in d}
oia_to_bdd_split=Counter()
for osplit,p,base,suf,exact,basep in all_oia:
    if basep:
        oia_to_bdd_split[(osplit, splitmap.get(base))]+=1
print('oia_split_to_bdd100k_split_by_base')
for k,v in sorted(oia_to_bdd_split.items()):
    print(k, v)

for root in [Path(r'E:/sbw/BDD100K'), repo/'dataset']:
    if root.exists():
        print('TREE', root)
        for p in root.iterdir():
            if p.is_dir():
                print(' ', p.name)
