import numpy as np, sys
sys.path.insert(0,'.')
from rupert import rid, PHI
def rid_z5():
    V=rid()
    a=np.array([0,PHI,1]); a/=np.linalg.norm(a)   # a 5-fold axis (through pentagon centres)
    z=np.array([0,0,1.0]); v=np.cross(a,z); s=np.linalg.norm(v); c=a@z
    K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    Rm=np.eye(3)+K+K@K*((1-c)/s**2)
    W=V@Rm.T
    return W/np.linalg.norm(W,axis=1).max()
if __name__=='__main__':
    W=rid_z5()
    def Rz(t): return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
    for ang,name in [(2*np.pi/5,'C5'),(np.pi,'C2?')]:
        U=W@Rz(ang).T
        print(name, np.min(np.linalg.norm(U[:,None]-W[None],axis=2),axis=1).max())
    print('central', np.min(np.linalg.norm(-W[:,None]-W[None],axis=2),axis=1).max())
