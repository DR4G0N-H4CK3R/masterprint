#!/usr/bin/env python3
"""Reproducible, offline partial-fingerprint security experiment."""
import argparse,csv,json,time,platform,hashlib
from pathlib import Path
import cv2,numpy as np,sklearn,joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from masterprint.data import load_socofing
from masterprint.features import extract,partials,match,detector_features
from masterprint.metrics import operating_point,binary_metrics,cluster_bootstrap

FEATURE_NAMES=['mean','std','q25','q50','q75','q90','max','top5_mean','contrast','sharpness','foreground','keypoint_fraction']
def savejson(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='data/SOCOFing');ap.add_argument('--n-identities',type=int,default=300)
    ap.add_argument('--matcher',choices=['minutiae','sift','orb'],default='minutiae')
    ap.add_argument('--target-fmr',type=float,default=.01);ap.add_argument('--crop-fraction',type=float,default=.70)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output',default='results/minutiae')
    ap.add_argument('--min-matches',type=int,default=2);ap.add_argument('--top-k',type=int,default=10)
    args=ap.parse_args()
    if args.n_identities<70:ap.error('Use at least 70 identities for seven disjoint groups')
    if not 0<args.target_fmr<1:ap.error('target-fmr must lie strictly between 0 and 1')
    if not .35<=args.crop_fraction<=.95:ap.error('crop-fraction must be 0.35 to 0.95')
    if args.min_matches<1 or args.top_k<1:ap.error('min-matches and top-k must be positive')
    start=time.time();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Remove optional outputs from earlier runs, so unavailable metrics cannot leave stale models.
    for name in ['detector.joblib','detector_roc.png']:
        (out/name).unlink(missing_ok=True)
    cv2.setNumThreads(1);np.random.seed(args.seed)
    images,manifest=load_socofing(args.data,args.n_identities)
    order=np.random.default_rng(args.seed).permutation(len(images));n=len(images);u=n//6
    splits=dict(calibration=order[:u],reference=order[u:2*u],label_gallery=order[2*u:3*u],target_gallery=order[3*u:4*u],train=order[4*u:5*u],validation=order[5*u:5*u+u//2],test=order[5*u+u//2:])
    role={int(i):k for k,v in splits.items() for i in v}
    for i,m in enumerate(manifest):m['split']=role[i]
    savejson(out/'manifest.json',manifest);savejson(out/'config.json',vars(args))
    print(f'Extracting {n*5} partial templates ({args.matcher})',flush=True)
    templates=[[extract(c,args.matcher) for c in partials(im,args.crop_fraction)] for im in images]
    # One held-out crop per calibration identity against 3 enrolled crops.
    cal=splits['calibration'];g=[];imp=[]
    for i in cal:
        for j in cal:
            s=max(match(templates[i][4],templates[j][k]) for k in range(3))
            (g if i==j else imp).append(s)
    op=operating_point(g,imp,args.target_fmr);threshold=op['threshold']
    print('Calibration:',op,flush=True)
    galleries={k:splits[k] for k in ['reference','label_gallery','target_gallery']}
    candidate_ids=np.r_[splits['train'],splits['validation'],splits['test']]
    # All candidate crops stay in the same person split.
    rows=[(int(i),c) for i in candidate_ids for c in range(5)]
    scores={k:np.zeros((len(rows),len(v))) for k,v in galleries.items()}
    X=[]
    for r,(i,c) in enumerate(rows):
        for name,ids in galleries.items():
            scores[name][r]=[max(match(templates[i][c],templates[j][k]) for k in range(3)) for j in ids]
        X.append(detector_features(scores['reference'][r],templates[i][c].quality))
        if r%50==0:print(f'Scored {r+1}/{len(rows)} candidates',flush=True)
    X=np.array(X);groups=np.array([i for i,c in rows]);roles=np.array([role[i] for i,c in rows])
    label_count=(scores['label_gallery']>=threshold).sum(axis=1)
    y=(label_count>=args.min_matches).astype(int)
    train=roles=='train';val=roles=='validation';test=roles=='test'
    model=None;prob=np.full(len(rows),.5);detector={'status':'unavailable: training labels contain only one class'}
    if len(np.unique(y[train]))==2:
        model=RandomForestClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',random_state=args.seed,n_jobs=1)
        model.fit(X[train],y[train]);prob=model.predict_proba(X)[:,list(model.classes_).index(1)]
        # Fit the detector operating point using validation negatives only.
        neg=prob[val & (y==0)]
        cut=float(np.nextafter(np.quantile(neg,.95,method='higher'),np.inf)) if len(neg) else .5
        detector=dict(status='trained',feature_names=FEATURE_NAMES,validation=binary_metrics(y[val],prob[val],cut),test=binary_metrics(y[test],prob[test],cut),feature_importance=dict(zip(FEATURE_NAMES,model.feature_importances_.tolist())))
        joblib.dump(dict(model=model,threshold=cut,features=FEATURE_NAMES,matcher=args.matcher),out/'detector.joblib')
    else:cut=.5
    discovery=label_count/len(galleries['label_gallery']);target=(scores['target_gallery']>=threshold).mean(axis=1)
    # Candidate selection uses discovery scores and detector only. Target scores never select a candidate.
    # Tune a finite selection penalty on validation, freeze it, then apply on test.
    penalties=[0,.05,.1,.2,.5,1.]
    def select(mask,penalty):
        ids=np.flatnonzero(mask);k=min(args.top_k,len(ids));return ids[np.argsort(-(discovery[ids]-penalty*prob[ids]),kind='stable')[:k]]
    baseval=select(val,0);floor=float(discovery[baseval].mean()*.8)
    feasible=[]
    for a in penalties:
        ids=select(val,a)
        if discovery[ids].mean()>=floor:feasible.append((float(prob[ids].mean()),a))
    penalty=min(feasible)[1] if feasible else 0.
    selection={}
    for name,ids in [('baseline',select(test,0)),('adaptive',select(test,penalty))]:
        accepted=(scores['target_gallery'][ids]>=threshold)
        selection[name]=dict(n_candidates=len(ids),discovery_coverage=float(discovery[ids].mean()),mean_target_coverage=float(target[ids].mean()),union_target_coverage=float(accepted.any(axis=0).mean()),flag_rate=float((prob[ids]>=cut).mean()) if model else None,mean_detector_probability=float(prob[ids].mean()) if model else None,candidate_rows=ids.tolist())
    report=dict(project='MasterPrint partial fingerprint research',dataset='SOCOFing Real',protocol='one left index finger per person; five 70%-side crops; same-image crop study',config=vars(args),environment=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,sklearn=sklearn.__version__),split_people={k:len(v) for k,v in splits.items()},calibration=op,labels=dict(definition=f'at least {args.min_matches} accepted identities in independent label gallery',train_positives=int(y[train].sum()),validation_positives=int(y[val].sum()),test_positives=int(y[test].sum())),detector=detector,selection=dict(penalty=penalty,validation_retention_floor=floor,results=selection),test_coverage=dict(mean=float(target[test].mean()),max=float(target[test].max()),person_bootstrap_95ci=cluster_bootstrap(target[test],groups[test],args.seed)),limitations=['Same-image crop overlap inflates genuine similarity; independent sensor captures are required for deployment claims.','Research matcher is not a commercial biometric SDK.','Label-gallery breadth is a proxy label, not a known physical MasterPrint.','Candidate crops and pair comparisons are correlated.','Adaptive evaluation selects existing crops only; no print generation or physical spoof test.','Low-FMR claims need larger calibration sets.'],elapsed_seconds=time.time()-start)
    savejson(out/'report.json',report)
    np.savez_compressed(out/'scores.npz',**scores,X=X,y=y,probabilities=prob,roles=roles,groups=groups,genuine=np.array(g),impostor=np.array(imp))
    with (out/'candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row','subject','crop','split','label_matches','label','target_coverage','detector_probability'])
        for r,(i,c) in enumerate(rows):w.writerow([r,manifest[i]['subject'],c,role[i],int(label_count[r]),int(y[r]),target[r],prob[r] if model else ''])
    plt.figure(figsize=(8,4));plt.hist(g,bins=25,alpha=.6,density=True,label='Genuine crop comparisons');plt.hist(imp,bins=25,alpha=.6,density=True,label='Impostor comparisons');plt.axvline(threshold,c='black',linestyle='--',label='Frozen threshold');plt.xlabel('Similarity');plt.ylabel('Density');plt.legend();plt.tight_layout();plt.savefig(out/'score_dist.png',dpi=160);plt.close()
    plt.figure(figsize=(8,4));plt.hist(target[test],bins=15,color='#1b8c83');plt.xlabel('Fraction of target identities accepted');plt.ylabel('Test candidates');plt.tight_layout();plt.savefig(out/'coverage.png',dpi=160);plt.close()
    if model and len(np.unique(y[test]))==2:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(y[test],prob[test]);plt.tight_layout();plt.savefig(out/'detector_roc.png',dpi=160);plt.close()
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
