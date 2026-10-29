from pathlib import Path
import re,hashlib
import cv2

def load_socofing(root,n=300):
    root=Path(root)
    real=root/'Real' if (root/'Real').is_dir() else root/'SOCOFing'/'Real'
    if not real.is_dir():raise ValueError(f'Expected {root}/Real or {root}/SOCOFing/Real')
    records=[]
    # Exactly one finger per person. This prevents cross-finger person leakage.
    for p in real.iterdir():
        m=re.fullmatch(r'(\d+)__[MF]_Left_index_finger\.BMP',p.name,re.I)
        if m:records.append((int(m[1]),p))
    records=sorted(records)[:n]
    if len(records)<n:raise ValueError(f'Requested {n} people, found {len(records)} left index fingers')
    images=[];manifest=[];seen=set()
    for subject,p in records:
        raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest in seen:raise ValueError('Duplicate image bytes across people')
        seen.add(digest)
        im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if im is None:raise ValueError(f'Cannot decode {p}')
        images.append(im);manifest.append(dict(subject=subject,file=p.name,sha256=digest,height=im.shape[0],width=im.shape[1]))
    return images,manifest
