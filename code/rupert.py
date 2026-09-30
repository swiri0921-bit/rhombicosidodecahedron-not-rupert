import numpy as np, itertools
from scipy.spatial import ConvexHull
from scipy.optimize import linprog, minimize
PHI=(1+5**0.5)/2
def even_perms(v):
    out=[]
    for p in itertools.permutations(range(3)):
        # parity
        inv=sum(1 for i in range(3) for j in range(i+1,3) if p[i]>p[j])
        if inv%2==0: out.append([v[p[0]],v[p[1]],v[p[2]]])
    return out
def signs(vs):
    out=set()
    for v in vs:
        for s in itertools.product([1,-1],repeat=3):
            out.add(tuple(round(a*b,12) for a,b in zip(v,s)))
    return np.array(sorted(out))
def cube(): return signs([[1,1,1]])
def rid():
    vs=[]
    for v in [(1,1,PHI**3),(PHI**2,PHI,2*PHI),(2+PHI,0,PHI**2)]: vs+=even_perms(v)
    return signs(vs)
def snub_cube():
    t=1.8392867552141612  # tribonacci constant
    vs=[]
    base=(1,1/t,t)
    for s in itertools.product([1,-1],repeat=3):
        if s[0]*s[1]*s[2]==1:  # even number of minus signs
            v=[a*b for a,b in zip(base,s)]; vs+=even_perms(v)
        else:
            v=[a*b for a,b in zip(base,s)]; vs+=[[p[0],p[1],p[2]] for p in odd_perms(v)]
    return np.unique(np.round(np.array(vs),12),axis=0)
def odd_perms(v):
    out=[]
    for p in itertools.permutations(range(3)):
        inv=sum(1 for i in range(3) for j in range(i+1,3) if p[i]>p[j])
        if inv%2==1: out.append([v[p[0]],v[p[1]],v[p[2]]])
    return out
def rot(a,b,c):
    # z-y-z Euler
    def Rz(t): return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
    def Ry(t): return np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])
    return Rz(a)@Ry(b)@Rz(c)
def proj(V,R): return (V@R.T)[:,:2]
def halfplanes(P2):
    h=ConvexHull(P2); return h.equations  # a x + b y + c <= 0
def mu(V,x):
    # x = (a1,b1, a2,b2,g): outer rot(0? ) ; inner rot + in-plane angle g
    Ro=rot(0,x[0],x[1]); Ri=rot(x[4],x[2],x[3])
    E=halfplanes(proj(V,Ro)); Q=proj(V,Ri); Q=Q[ConvexHull(Q).vertices]
    A=E[:,:2]; c=E[:,2]
    # constraints: A (s q + t) + c <= 0  ->  (A q) s + A t <= -c
    rows=[]; rhs=[]
    for q in Q:
        rows.append(np.column_stack([A@q, A])); rhs.append(-c)
    Aub=np.vstack(rows); bub=np.concatenate(rhs)
    r=linprog([-1,0,0],A_ub=Aub,b_ub=bub,bounds=[(0,None),(None,None),(None,None)],method='highs')
    return r.x[0] if r.status==0 else 0.0
