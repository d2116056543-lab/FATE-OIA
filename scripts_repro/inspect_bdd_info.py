import json
from pathlib import Path
p=Path(r'E:/sbw/BDD100K/bdd100k_info/bdd100k/info/100k/train/000f157f-dab3a407.json')
data=json.loads(p.read_text())
print(type(data), data.keys())
for k,v in data.items():
    if isinstance(v,(str,int,float,bool,type(None))): print(k, v)
    elif isinstance(v, list): print(k, 'list', len(v), 'first', v[0] if v else None)
    elif isinstance(v, dict): print(k, 'dict keys', list(v.keys())[:20])
