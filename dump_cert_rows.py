"""Dump the combinatorial and decimal-derived data of every sector and tube certificate (contact rows, active
sets, exact-partner choices, D-values, number of exact claims).  Used by verify_decimal_precision.py."""
import sys, json, numpy as np
sys.path.insert(0,'.')
from decimal import Decimal as D
from dec_util import mat
out={}
from rig_tube import RigTube, EX
for which,ref in (('C','ref1'),('C','ref2'),('D','ref1')):
    Tm=EX['T'+which]; Mfix=None
    if ref=='ref2':
        Tm=mat(EX['TC'],EX['h2C']); Mfix=[[D(-1),D(0),D(0)],[D(0),D(-1),D(0)],[D(0),D(0),D(1)]]
    T=RigTube(which,Tm,1 if which=='C' else -1,Mfix=Mfix,r0=1e-3,eps0=0.4,beta=0.5)
    out['tube_'+which+ref]=[ [c['meta'][0],c['meta'][3],c['meta'][4],c['meta'][7],c['Dp'],c['Dm'],c['Dp0'],c['Dm0']] for c in T.C]
    out['claims_'+which+ref]=len(T.claims)
from rig_sector import RigSector
RZ=[[D(-1),D(0),D(0)],[D(0),D(-1),D(0)],[D(0),D(0),D(1)]]
permA=list(np.load('./permA.npy'))
for name,mk in (('A1',lambda: RigSector('A',mode='xi',r0=1.5e-3)),('A2',lambda: RigSector('A',perm=permA,Mfix=RZ,mode='xi',r0=1.5e-3)),
                ('B1',lambda: RigSector('B',mode='xi',r0=1.5e-3)),('C1',lambda: RigSector('C',Vin_mat=EX['TC'],mode='s',r0=1e-3)),
                ('D1',lambda: RigSector('D',Vin_mat=EX['TD'],mode='s',r0=1e-3))):
    R=mk(); out['sec_'+name]=[[c['meta'][0],c['meta'][3],c['meta'][4],c['meta'][5],c['Z']] for c in R.C]
json.dump(out,open(sys.argv[1],'w'),default=str)
print({k:(len(v) if isinstance(v,list) else v) for k,v in out.items()})
