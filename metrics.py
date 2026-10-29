import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix

def operating_point(genuine,impostor,target):
    g=np.asarray(genuine);i=np.asarray(impostor)
    if not len(g) or not len(i):raise ValueError('Calibration requires genuine and impostor scores')
    # Acceptance uses >=. Enumerate unique thresholds plus a strict upper endpoint.
    thresholds=np.r_[np.unique(np.r_[g,i]),np.nextafter(max(g.max(),i.max()),np.inf)]
    fmr=np.array([(i>=t).mean() for t in thresholds]);fnmr=np.array([(g<t).mean() for t in thresholds])
    idx=np.flatnonzero(fmr<=target)[0];ei=np.argmin(abs(fmr-fnmr))
    return dict(threshold=float(thresholds[idx]),target_fmr=target,calibration_fmr=float(fmr[idx]),calibration_fnmr=float(fnmr[idx]),eer=float((fmr[ei]+fnmr[ei])/2),eer_threshold=float(thresholds[ei]),n_genuine=len(g),n_impostor=len(i))

def binary_metrics(y,p,threshold=.5):
    y=np.asarray(y);p=np.asarray(p);pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),positives=int(y.sum()),negatives=int((y==0).sum()),tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),tpr=float(tp/(tp+fn)) if tp+fn else None,fpr=float(fp/(fp+tn)) if fp+tn else None,auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,threshold=float(threshold))

def cluster_bootstrap(values,groups,seed=42,n=1000):
    rng=np.random.default_rng(seed);groups=np.asarray(groups);values=np.asarray(values)
    means=np.array([values[groups==g].mean() for g in np.unique(groups)])
    draws=means[rng.integers(0,len(means),(n,len(means)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws,[.025,.975])]
