import numpy as np, sys, time, itertools, json, pickle
sys.path.insert(0,'.')
from rig_tube import RigTube, EX
from decimal import Decimal as D
from dec_util import mat
which=sys.argv[1]; ref=sys.argv[2]; r0=float(sys.argv[3]); eps0=float(sys.argv[4]); tlim=float(sys.argv[5]); beta=float(sys.argv[6]) if len(sys.argv)>6 else 0.5
sgn = 1 if which=='C' else -1
Tm = EX['T'+which]; Mfix=None
if ref=='ref2':
    Tm = mat(EX['TC'], EX['h2C']); Mfix=[[D(-1),D(0),D(0)],[D(0),D(-1),D(0)],[D(0),D(0),D(1)]]
t0=time.time(); T=RigTube(which,Tm,sgn,Mfix=Mfix,r0=r0,eps0=eps0,beta=beta); print(which,ref,'rig tube contacts',len(T.C),'build',round(time.time()-t0,1),flush=True)
from exact_verify import verify
Tq=pickle.load(open(f'./twist_body_{which}.pkl','rb'))
if ref=='ref2':
    from exact_check import mat_from_perm, V as VQ
    import numpy as _np
    # h2 (z5 float) -> exact body symmetry via vertex permutation
    from rid_setup import rid_z5
    PFz=rid_z5(); h2=_np.array(json.load(open('./twist_pts.json'))['C']['h2'])
    perm=[int(_np.argmin(_np.linalg.norm(PFz-(h2@x),axis=1))) for x in PFz]
    Hq=mat_from_perm(perm); Tq=[[sum((Tq[i][k]*Hq[k][j] for k in range(3)),__import__('q5').Q5(0)) for j in range(3)] for i in range(3)]
    badc=verify(which,T.claims,Tq=Tq,smul=-1)
else:
    badc=verify(which,T.claims,Tq=Tq)
print('exact claims',len(T.claims),'violations',len(badc),flush=True)
assert not badc
def omega_box(face,lo,hi):
    a,sg=face; c=(lo+hi)/2; v=np.insert(c,a,sg); return v/np.linalg.norm(v), float(np.linalg.norm((hi-lo)/2))*(1+1e-12) + 1e-15   # abs. slack: float centre
import os
CK=f'./rig/tube_{which}_{ref}_{eps0}_{beta}.ckpt'
if os.path.exists(CK):
    stack, ok, fail = pickle.load(open(CK, 'rb')); print('resumed from checkpoint', len(stack), ok, flush=True)
else:
    stack=[((a,sg),np.array([-1.,-1,0]),np.array([1.,1,1]),sigma,0) for a in range(3) for sg in (-1,1) for sigma in (-1,1)]
    ok=0; fail=[]
t0=time.time(); tck=time.time()
while stack:
    if time.time()-tck > 60:
        pickle.dump((stack, ok, fail), open(CK+'.tmp', 'wb')); os.replace(CK+'.tmp', CK); tck=time.time()
    face,lo,hi,sigma,dp=stack.pop()
    oc,orad=omega_box(face,lo[:2],hi[:2])
    if T.certify(oc,orad,lo[2],hi[2],sigma): ok+=1; continue
    if dp>=16 or time.time()-t0>tlim: fail.append((face,lo.tolist(),hi.tolist(),sigma)); continue
    w=hi-lo; sc=w/np.array([2,2,1]); dims=[d for d in range(3) if sc[d]>=0.6*sc.max()]; mid=(lo+hi)/2
    for bits in itertools.product([0,1],repeat=len(dims)):
        nl=lo.copy(); nh=hi.copy()
        for d,b in zip(dims,bits):
            if b: nl[d]=mid[d]
            else: nh[d]=mid[d]
        stack.append((face,nl,nh,sigma,dp+1))
print(which,ref,'RIGOROUS tube ok',ok,'fail',len(fail),'time',round(time.time()-t0,1),flush=True)
json.dump(fail,open(f'./rig/tube_{which}_{ref}_{eps0}_{beta}.json','w'))
if os.path.exists(CK): os.remove(CK)
