"""Unify supplied NPY and MAT labels by emotion name, preserving every class."""
from pathlib import Path
import argparse, hashlib, json, shutil
import h5py
import hdf5storage
import numpy as np
from scipy.io import loadmat, savemat

CANONICAL = dict(happy=0, sad=1, angry=2, neutral=3, surprise=4, fear=5, disgust=6)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def remap(labels, label_map):
    values=np.asarray(labels)
    unknown=set(np.unique(values))-set(label_map.values())
    if unknown:raise ValueError(f'Labels absent from mapping: {unknown}')
    result=np.empty(values.shape,dtype=values.dtype)
    for emotion,old in label_map.items():
        if emotion not in CANONICAL:raise ValueError(f'Unsupported emotion: {emotion}')
        result[values==old]=CANONICAL[emotion]
    return result

def unify(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or source in destination.parents:
        raise ValueError('Output must be a separate directory')
    if destination.exists() and any(destination.iterdir()):raise FileExistsError(destination)
    destination.mkdir(parents=True,exist_ok=True)
    report=[]
    for domain in ['C','E','M','Me']:
        src=source/domain;out=destination/domain;out.mkdir()
        mapping=json.loads((src/'label_map.json').read_text(encoding='utf-8'))
        labels=np.load(src/'labels.npy',allow_pickle=False)
        x=np.load(src/'features.npy',mmap_mode='r',allow_pickle=False)
        if len(x)!=len(labels):raise ValueError(f'{domain}: row mismatch')
        updated=remap(labels,mapping)
        shutil.copy2(src/'features.npy',out/'features.npy')
        np.save(out/'labels.npy',updated)
        (out/'label_map.json').write_text(json.dumps(CANONICAL,indent=2),encoding='utf-8')
        for p in src.iterdir():
            if p.is_file() and p.name not in ['features.npy','labels.npy','label_map.json']:
                shutil.copy2(p,out/p.name)
        feature_hash=digest(src/'features.npy')
        assert feature_hash==digest(out/'features.npy')
        r={'domain':domain,'shape':list(x.shape),'original_map':mapping,'unified_map':CANONICAL,'emotion_counts':{emotion:int((labels==old).sum()) for emotion,old in mapping.items()},'features_sha256':feature_hash,'original_labels_sha256':digest(src/'labels.npy'),'unified_labels_sha256':digest(out/'labels.npy'),'sample_order':'unchanged'}
        # Regenerate a MATLAB v7.3 file from the same NPY rows: historical MAT
        # artifacts can use a different ordering and even a different sample count.
        mat=source/(domain+'.mat')
        if mat.exists():
            if h5py.is_hdf5(mat):
                with h5py.File(mat,'r') as f:
                    key=next((k for k in ['labels','label','y','Y'] if k in f),None)
                    if key is None:raise ValueError(f'{mat}: labels missing')
                    r['original_MAT_label_count']=int(f[key].size)
            else:
                content=loadmat(mat)
                key=next((k for k in ['labels','label','y','Y'] if k in content),None)
                if key is None:raise ValueError(f'{mat}: labels missing')
                r['original_MAT_label_count']=int(content[key].size)
        reverse={v:k for k,v in CANONICAL.items()}
        label_names=np.array([reverse[int(v)] for v in updated],dtype=object).reshape(-1,1)
        dst=destination/(domain+'.mat')
        hdf5storage.savemat(str(dst),{'features':np.asarray(x),'labels':updated.reshape(-1,1),'labels_str':label_names,'label_map':CANONICAL},format='7.3',store_python_metadata=False)
        with h5py.File(dst,'r') as f:
            assert np.array_equal(f['labels'][()].reshape(-1),updated)
            assert np.array_equal(f['features'][()].T,x)
        r['MAT_status']='Regenerated MATLAB v7.3 from unified NPY; same features, labels and sample order'
        r['unified_MAT_sha256']=digest(dst)
        report.append(r);print(domain,r['shape'],r['emotion_counts'],flush=True)
    (destination/'label-unification-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();unify(a.input,a.output)
