#!/usr/bin/env python3
"""Score a local image against the frozen research reference gallery."""
import argparse,json
from pathlib import Path
import cv2,joblib,numpy as np
from masterprint.features import extract,partials,match,detector_features
p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--result',default='results/minutiae');p.add_argument('--data',default='data/SOCOFing/Real');a=p.parse_args()
r=Path(a.result)
if not (r/'detector.joblib').is_file():raise SystemExit('This run has no trained detector. See report.json for the reason.')
bundle=joblib.load(r/'detector.joblib') # Load only your trusted project model files.
config=json.loads((r/'config.json').read_text());manifest=json.loads((r/'manifest.json').read_text());cv2.setNumThreads(1)
im=cv2.imread(a.image,0)
if im is None:raise SystemExit('Cannot read input image')
t=extract(im,bundle['matcher']);scores=[]
for m in manifest:
 if m['split']=='reference':
  ref=cv2.imread(str(Path(a.data)/m['file']),0)
  if ref is None:raise SystemExit(f"Missing reference: {m['file']}")
  enrolled=[extract(c,bundle['matcher']) for c in partials(ref,config['crop_fraction'])[:3]]
  scores.append(max(match(t,e) for e in enrolled))
x=detector_features(np.asarray(scores),t.quality).reshape(1,-1)
prob=float(bundle['model'].predict_proba(x)[0,list(bundle['model'].classes_).index(1)])
print(json.dumps(dict(risk_probability=prob,threshold=bundle['threshold'],flagged=bool(prob>=bundle['threshold']),interpretation='Research proxy for breadth of matching. Not a liveness or authentication decision.'),indent=2))
