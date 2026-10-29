"""CPU research templates. Scores are empirical similarities, not SDK scores."""
from dataclasses import dataclass
import cv2
import numpy as np
from skimage.morphology import skeletonize

@dataclass
class Template:
    points: np.ndarray
    descriptors: np.ndarray | None
    quality: np.ndarray
    binary: bool = False

def partials(image, fraction=.70):
    h,w=image.shape
    ph,pw=int(h*fraction),int(w*fraction)
    if min(ph,pw)<24: raise ValueError('Partial crops must be at least 24 pixels')
    return [image[y:y+ph,x:x+pw].copy() for y,x in [(0,0),(0,w-pw),(h-ph,0),(h-ph,w-pw),((h-ph)//2,(w-pw)//2)]]

def extract(image, matcher='minutiae'):
    # Magnification improves descriptor support but adds no sensor information.
    im=cv2.resize(image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    im=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(im)
    quality=np.array([im.std()/128,cv2.Laplacian(im,cv2.CV_32F).var()/10000, float((im<220).mean())],dtype=float)
    if matcher=='orb':
        kp,des=cv2.ORB_create(nfeatures=100,edgeThreshold=8,patchSize=21,fastThreshold=7).detectAndCompute(im,None)
    elif matcher=='sift':
        kp,des=cv2.SIFT_create(nfeatures=100,contrastThreshold=.015,edgeThreshold=15).detectAndCompute(im,None)
    else:
        # Ridge endings/bifurcations from crossing number on a thinned ridge map.
        ridge=cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,4)>0
        sk=skeletonize(ridge).astype(np.uint8)
        ns=[np.roll(np.roll(sk,dy,0),dx,1) for dy,dx in [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]]
        cn=sum(np.abs(ns[i].astype(int)-ns[(i+1)%8].astype(int)) for i in range(8))/2
        ys,xs=np.where((sk>0)&((cn==1)|(cn==3)))
        kp=[]
        gx=cv2.Sobel(im,cv2.CV_32F,1,0);gy=cv2.Sobel(im,cv2.CV_32F,0,1)
        for y,x in zip(ys,xs):
            if 15<x<im.shape[1]-15 and 15<y<im.shape[0]-15 and len(kp)<100:
                if all((x-k.pt[0])**2+(y-k.pt[1])**2>36 for k in kp):
                    # Local ridge orientation from the gradient structure tensor.
                    a=gx[y-4:y+5,x-4:x+5];b=gy[y-4:y+5,x-4:x+5]
                    angle=(np.degrees(.5*np.arctan2(2*(a*b).sum(),(a*a-b*b).sum()))+90)%360
                    kp.append(cv2.KeyPoint(float(x),float(y),16,float(angle)))
        kp,des=cv2.SIFT_create().compute(im,kp)
    pts=np.array([k.pt for k in kp],dtype=np.float32).reshape(-1,2)
    return Template(pts,des,np.r_[quality,len(pts)/100],matcher=='orb')

def match(a,b):
    if a.descriptors is None or b.descriptors is None or min(len(a.points),len(b.points))<4:return 0.
    bf=cv2.BFMatcher(cv2.NORM_HAMMING if a.binary else cv2.NORM_L2)
    pairs=bf.knnMatch(a.descriptors,b.descriptors,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.78*n.distance]
    # Require unique endpoints to avoid inflated matches on repeated ridge texture.
    unique={}
    for m in sorted(good,key=lambda z:z.distance):unique.setdefault(m.trainIdx,m)
    good=list(unique.values())
    if len(good)<4:return 0.
    src=np.float32([a.points[m.queryIdx] for m in good]);dst=np.float32([b.points[m.trainIdx] for m in good])
    cv2.setRNGSeed(123)
    mat,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=200,confidence=.99,refineIters=5)
    if mat is None or inliers is None:return 0.
    scale=float(np.hypot(mat[0,0],mat[1,0]))
    if not .8<=scale<=1.25:return 0.
    count=int(inliers.sum())
    return count/min(len(a.points),len(b.points)) if count>=4 else 0.

def detector_features(scores, quality):
    s=np.sort(scores)
    return np.r_[s.mean(),s.std(),np.quantile(s,[.25,.5,.75,.9]),s[-1],np.mean(s[-min(5,len(s)):]),quality]
