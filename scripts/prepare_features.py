"""Create a canonical, separate copy; original files and sample order are preserved."""
from pathlib import Path
import argparse, json
import numpy as np

KNOWN = ['happy', 'sad', 'angry']
def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Use a separate output directory outside the original feature directory')
    for domain in ['C','E','M','Me']:
        src=source/domain; out=destination/domain
        x=np.load(src/'features.npy',mmap_mode='r',allow_pickle=False)
        y=np.load(src/'labels.npy',allow_pickle=False)
        label_map=json.loads((src/'label_map.json').read_text(encoding='utf-8'))
        if x.ndim!=2 or len(x)!=len(y) or y.ndim!=1:
            raise ValueError(f'{domain}: invalid shapes {x.shape}, {y.shape}')
        missing=set(np.unique(y))-set(label_map.values())
        if missing: raise ValueError(f'{domain}: unmapped labels {missing}')
        if not all(k in label_map for k in KNOWN): raise ValueError(f'{domain}: missing known emotions')
        mapped=np.full(y.shape,3,dtype=np.int64)
        for new,emotion in enumerate(KNOWN): mapped[y==label_map[emotion]]=new
        out.mkdir(parents=True,exist_ok=True)
        if (out/'features.npy').exists(): raise FileExistsError(out/'features.npy')
        np.save(out/'features.npy',x); np.save(out/'labels.npy',mapped)
        (out/'label_map.json').write_text(json.dumps(dict(happy=0,sad=1,angry=2,unknown=3),indent=2),encoding='utf-8')
        (out/'provenance.json').write_text(json.dumps({'source':str(src),'original_label_map':label_map,'shape':list(x.shape),'sample_order':'unchanged','splits':'not inferred'},indent=2),encoding='utf-8')
        print(domain,x.shape,'canonical labels',np.unique(mapped,return_counts=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.input,a.output)
