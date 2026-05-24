from pathlib import Path
from PIL import Image
import json, random, re
repo=Path(r'E:/sbw/SNNA_repro/SNNA')
oia=repo/'dataset/BDD-OIA'
bdd=Path(r'E:/sbw/BDD100K/bdd100k_images/bdd100k/images/100k')

def strip_suffix(name):
    m=re.match(r'^(.*)_([0-9]+)(\.jpg)$', name)
    return (m.group(1)+m.group(3), int(m.group(2))) if m else (name,None)

pairs=[]
for split in ['train','val','test']:
    for p in (oia/split).glob('*.jpg'):
        base,s=strip_suffix(p.name)
        bp=next((bdd/sp/base for sp in ['train','val','test'] if (bdd/sp/base).exists()), None)
        if bp: pairs.append((split,p,bp,s))
print('pairs',len(pairs))
for split,p,bp,s in random.sample(pairs,5):
    with Image.open(p) as im1, Image.open(bp) as im2:
        print('sample', split, p.name, 'suffix', s, 'oia_size', im1.size, 'bdd_base', bp.parent.name+'/'+bp.name, 'bdd_size', im2.size, 'file_sizes', p.stat().st_size, bp.stat().st_size)
for js in ['train.json','val.json','test.json']:
    path=oia/js
    data=json.loads(path.read_text(encoding='utf-8'))
    print('json', js, 'type', type(data).__name__, 'len', len(data) if hasattr(data,'__len__') else 'na')
    # print first few keys/items compactly
    if isinstance(data, dict):
        keys=list(data.keys())[:5]
        print(' keys', keys)
        for k in keys[:2]: print(' item', k, data[k])
    elif isinstance(data, list):
        for item in data[:2]: print(' item', item)
