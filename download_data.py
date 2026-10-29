#!/usr/bin/env python3
"""Download original SOCOFing images from the authors' Kaggle distribution."""
import argparse,hashlib,json,shutil,tempfile,urllib.request,zipfile
from pathlib import Path
URL='https://www.kaggle.com/api/v1/datasets/download/ruizgara/socofing'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='data/SOCOFing');args=ap.parse_args()
    dest=Path(args.output)/'Real';dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        archive=Path(temp)/'socofing.zip'
        print('Downloading approximately 878 MB. Only Real/ images will be retained.',flush=True)
        with urllib.request.urlopen(URL,timeout=120) as src,archive.open('wb') as f:shutil.copyfileobj(src,f)
        count=0
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if '/Real/' in name and name.lower().endswith('.bmp'):
                    (dest/Path(name).name).write_bytes(z.read(name));count+=1
        if count!=6000:raise RuntimeError(f'Expected 6000 originals, found {count}')
        print(f'Saved {count} original images in {dest}')
if __name__=='__main__':main()
