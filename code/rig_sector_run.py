import numpy as np, sys, time, itertools, json, os
sys.path.insert(0,'.')
from rig_sector import RigSector, EX
from decimal import Decimal as D
from dec_util import mat
which, ref, mode, piece, face, r0, tlim = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
TAG = os.environ.get('TAG', ''); out = f'./rig/sec_{which}_{ref}_{mode}_{piece}_{face}{TAG}.json'
if os.path.exists(out): sys.exit(0)
RZ = [[D(-1),D(0),D(0)],[D(0),D(-1),D(0)],[D(0),D(0),D(1)]]
DPSI = json.load(open('./dpsi_exact.json'))
psi_arc = {'C': np.pi/2, 'D': -np.pi/2}.get(which)
kw = dict(mode=mode, r0=r0)
if which in ('A','B'):
    if ref == 'ref1': mk = lambda **k: RigSector(which, **kw, **k)
    else:
        permA = list(np.load('./permA.npy')); mk = lambda **k: RigSector(which, perm=permA, Mfix=RZ, **kw, **k)
else:
    Tm = EX['T'+which]; mk = lambda **k: RigSector(which, Vin_mat=Tm, **kw, **k)
RS = [mk()]
if mode == 's':
    RS.append(mk(fan=np.array([0.0]), chifan=np.round(np.arange(-3,3.0001,0.05),10), chidir=0))
    RS.append(mk(fan=np.round(np.arange(-3,3.0001,0.025),10)))
print(which, ref, mode, piece, face, 'contacts', [len(r.C) for r in RS], flush=True)
def angd(a,b): return abs((a-b+np.pi)%(2*np.pi)-np.pi)
def skip(oc, orad, wl, wh, tl, th, lo, hi):
    if which == 'A' and ref == 'ref1':
        Dp = np.array(DPSI['A']); wc=(wl+wh)/2; wr=np.linalg.norm((wh-wl)/2); tc=(tl+th)/2
        wmin = np.linalg.norm(np.clip(0, wl, wh)); wmax = np.linalg.norm(np.maximum(np.abs(wl), np.abs(wh)))
        b = np.linalg.norm(tc*oc - Dp@wc) + th*orad + (th-tl)/2 + np.linalg.norm(Dp,2)*wr + 0.02*wmax + 1e-9
        return b <= 0.6*wmin
    if which in ('C','D') and mode == 's' and piece == 2:
        # tube1: |cot psi| <= 0.4, that <= 0.5 sin psi  (guaranteed with margins)
        if angd(lo[2], psi_arc) <= 0.378 and angd(hi[2], psi_arc) <= 0.378 and hi[2]-lo[2] < np.pi and hi[3] <= 0.46: return True
    if which == 'C' and mode == 's':
        Dp = np.array(DPSI['C']); SG = np.sign(np.sin(psi_arc))
        hmin = min(SG*wl[1], SG*wh[1])
        if hmin <= 0: return False
        if max(abs(wl[0]), abs(wh[0])) > 0.39*hmin: return False
        wc=(wl+wh)/2; wr=np.linalg.norm((wh-wl)/2); tc=(tl+th)/2; wmax=np.linalg.norm(np.maximum(np.abs(wl),np.abs(wh)))
        b = np.linalg.norm(tc*oc - Dp@wc) + th*orad + (th-tl)/2 + np.linalg.norm(Dp,2)*wr + 0.02*wmax + 1e-9
        return b <= 0.49*hmin
    return False
def omega_box(fc, lo, hi):
    a,sg=fc; c=(lo+hi)/2; v=np.insert(c,a,sg); return v/np.linalg.norm(v), float(np.linalg.norm((hi-lo)/2))*(1+1e-12) + 1e-15   # abs. slack: float centre
a_, sg_ = face//2, (face%2)*2-1
thmax = 0.8 if (which=='A' and ref=='ref2') else 1.0
MAXDP = max(int(os.environ.get('MAXDP', '18')), 28)   # deep point-like spots: children certify at depth 23-24
if os.environ.get('INITFILE'):
    stack=[(tuple(b[0]),np.array(b[1],float),np.array(b[2],float),int(b[3]) if len(b)>3 else 0) for b in json.load(open(os.environ['INITFILE']))]
elif os.environ.get('INITBOX'):
    ib=json.loads(os.environ['INITBOX']); stack=[(tuple(ib[0]),np.array(ib[1],float),np.array(ib[2],float),0)]
elif piece == 1: stack=[((a_,sg_),np.array([-1.,-1,-1,-1]),np.array([1.,1,1,1]),0)]
else: stack=[((a_,sg_),np.array([-1.,-1,0,0]),np.array([1.,1,2*np.pi,thmax]),0)]
ok=0; sk=0; fail=[]; t0=time.time()
while stack:
    fc,lo,hi,dp=stack.pop()
    oc,orad=omega_box(fc,lo[:2],hi[:2])
    if piece==1:
        wl,wh=lo[2:].copy(),hi[2:].copy()
        if np.linalg.norm(np.clip(0,wl,wh))>1: ok+=1; continue
        tl=th=1.0
    else:
        ps=np.linspace(lo[2],hi[2],9); pts=np.stack([np.cos(ps),np.sin(ps)],1); wl=pts.min(0)-1e-9; wh=pts.max(0)+1e-9
        for k in range(-2,6):
            ang=k*np.pi/2
            if lo[2]<=ang<=hi[2]:
                dim=k%2; val=np.cos(ang) if dim==0 else np.sin(ang)
                if val>0: wh[dim]=1
                else: wl[dim]=-1
        tl,th=lo[3],hi[3]
    if skip(oc,orad,wl,wh,tl,th,lo,hi): sk+=1; continue
    c=False
    for j,R in enumerate(RS):
        if j>0 and dp<4: break
        if R.certify(oc,orad,wl,wh,tl,th): c=True; break
    if c: ok+=1; continue
    if dp>=MAXDP or time.time()-t0>tlim: fail.append((list(fc),lo.tolist(),hi.tolist(),dp)); continue
    w=hi-lo; sc=w/np.array([2,2,2*np.pi if piece==2 else 2,1 if piece==2 else 2]); dims=[d for d in range(4) if sc[d]>=0.6*sc.max()]; mid=(lo+hi)/2
    for bits in itertools.product([0,1],repeat=len(dims)):
        nl=lo.copy(); nh=hi.copy()
        for d,b in zip(dims,bits):
            if b: nl[d]=mid[d]
            else: nh[d]=mid[d]
        stack.append((fc,nl,nh,dp+1))
json.dump({'ok':ok,'skip':sk,'fail':fail,'time':time.time()-t0},open(out,'w'))
print('done ok',ok,'skip',sk,'fail',len(fail),'time',round(time.time()-t0,1),flush=True)
