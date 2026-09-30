"""Exact-ish (50-digit Decimal) rhombicosidodecahedron vertices in the z5 frame, and rigorous float enclosures."""
from decimal import Decimal as D, getcontext
import numpy as np, sys
sys.path.insert(0,'.')
getcontext().prec = 60
def even_perms(v):
    a,b,c=v; out=[]
    for (x,y,z) in [(a,b,c),(b,c,a),(c,a,b)]:
        for sx in (1,-1):
            for sy in (1,-1):
                for sz in (1,-1):
                    p=(sx*x,sy*y,sz*z)
                    if p not in out: out.append(p)
    return out
def rid_dec():
    s5=D(5).sqrt(); P=(1+s5)/2
    vs=[]
    for v in [(D(1),D(1),P**3),(P**2,P,2*P),(2+P,D(0),P**2)]: vs+=even_perms(v)
    # dedupe (zeros)
    U=[]
    for v in vs:
        if not any(all(abs(v[i]-u[i])<D('1e-40') for i in range(3)) for u in U): U.append(v)
    V=U
    a=[D(0),P,D(1)]; na=(a[1]**2+a[2]**2).sqrt(); a=[x/na for x in a]
    z=[D(0),D(0),D(1)]
    v=[a[1]*z[2]-a[2]*z[1], a[2]*z[0]-a[0]*z[2], a[0]*z[1]-a[1]*z[0]]
    s2=v[0]**2+v[1]**2+v[2]**2; c=a[2]
    K=[[D(0),-v[2],v[1]],[v[2],D(0),-v[0]],[-v[1],v[0],D(0)]]
    KK=[[sum(K[i][k]*K[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    f=(1-c)/s2
    R=[[(D(1) if i==j else D(0))+K[i][j]+KK[i][j]*f for j in range(3)] for i in range(3)]
    W=[[sum(R[i][k]*p[k] for k in range(3)) for i in range(3)] for p in V]
    r=max((w[0]**2+w[1]**2+w[2]**2).sqrt() for w in W)
    return [[x/r for x in w] for w in W]
if __name__=='__main__':
    from rid_setup import rid_z5
    Vf=rid_z5(); Wd=rid_dec()
    Wd_f=np.array([[float(x) for x in w] for w in Wd])
    # match order
    idx=[int(np.argmin(np.linalg.norm(Wd_f-v,axis=1))) for v in Vf]
    assert len(set(idx))==60
    err=max(abs(D(repr(float(Vf[i][j])))-Wd[idx[i]][j]) for i in range(60) for j in range(3))
    print('max |float - exact| =',float(err))
    np.save('./vert_delta.npy',np.array([float(err)*4+1e-300]))
