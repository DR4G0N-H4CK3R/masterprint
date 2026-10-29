#!/usr/bin/env python3
"""Check result provenance, disjoint people, and saved metric arithmetic."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from masterprint.metrics import binary_metrics
p=argparse.ArgumentParser();p.add_argument('result',nargs='?',default='results/minutiae');p.add_argument('--data',default='data/SOCOFing/Real');a=p.parse_args()
r=Path(a.result);report=json.loads((r/'report.json').read_text());manifest=json.loads((r/'manifest.json').read_text());s=np.load(r/'scores.npz')
assert len({m['subject'] for m in manifest})==len(manifest)
assert sum(report['split_people'].values())==len(manifest)
for m in manifest:assert hashlib.sha256((Path(a.data)/m['file']).read_bytes()).hexdigest()==m['sha256']
for role in ['train','validation','test']:
 ids=set(s['groups'][s['roles']==role])
 for other in ['train','validation','test']:
  if role!=other:assert ids.isdisjoint(s['groups'][s['roles']==other])
threshold=report['calibration']['threshold']
assert np.isclose((s['impostor']>=threshold).mean(),report['calibration']['calibration_fmr'])
if report['detector']['status']=='trained':
 d=report['detector']['test'];mask=s['roles']=='test'
 assert binary_metrics(s['y'][mask],s['probabilities'][mask],d['threshold'])==d
for name,selection in report['selection']['results'].items():
 ids=selection['candidate_rows'];cov=(s['target_gallery'][ids]>=threshold).mean(axis=1)
 assert np.isclose(cov.mean(),selection['mean_target_coverage'])
print('PASS: hashes, person separation, calibration, detector metrics and selection arithmetic')
