"""Verify downloaded release archives against SHA256SUMS.txt."""
from pathlib import Path
import argparse, hashlib, zipfile
def verify(folder,checksums='SHA256SUMS-active.txt'):
    folder=Path(folder);count=0
    for line in (folder/checksums).read_text().splitlines():
        expected,name=line.split(None,1);p=folder/name.strip()
        if not p.is_file():raise FileNotFoundError(p)
        h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(8*1024*1024):h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError('Checksum mismatch: '+name)
        with zipfile.ZipFile(p) as z:
            bad=z.testzip()
            if bad:raise ValueError(f'{name}: damaged entry {bad}')
        count+=1;print('OK',name)
    return count
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');p.add_argument('--checksums',default='SHA256SUMS-active.txt');a=p.parse_args();verify(a.folder,a.checksums)
