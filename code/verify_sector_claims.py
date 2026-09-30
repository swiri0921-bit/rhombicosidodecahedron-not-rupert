"""Exact Q(sqrt5) verification of all structural claims (K, KI, Kx, KIx, Z, ID) made by the RigSector objects used in the runs."""
import numpy as np, pickle, sys
sys.path.insert(0, '.')
from rig_sector import RigSector, EX
from exact_verify import verify
from decimal import Decimal as D
RZ = [[D(-1),D(0),D(0)],[D(0),D(-1),D(0)],[D(0),D(0),D(1)]]
permA = list(np.load('./permA.npy'))
cfg = [('A', 'ref1', 'xi', dict(), dict()),
       ('A', 'ref2', 'xi', dict(perm=permA, Mfix=RZ), dict(pm=permA, smul=-1)),
       ('B', 'ref1', 'xi', dict(), dict())]
for w in 'CD':
    cfg.append((w, 'ref1', 's', dict(Vin_mat=EX['T' + w]), dict(Tq=pickle.load(open(f'./twist_body_{w}.pkl', 'rb')))))
tot = 0
for which, ref, mode, kw, vk in cfg:
    variants = [dict()]
    if mode == 's': variants += [dict(fan=np.array([0.0]), chifan=np.round(np.arange(-3, 3.0001, 0.05), 10), chidir=0), dict(fan=np.round(np.arange(-3, 3.0001, 0.025), 10))]
    cl = set()
    for v in variants:
        R = RigSector(which, mode=mode, r0=1.5e-3 if which in 'AB' else 1e-3, **kw, **v); cl |= set(map(repr, R.claims))
    claims = [eval(c) for c in cl]
    bad = verify(which, claims, **vk)
    print(which, ref, 'claims', len(claims), 'types', sorted(set(c[0] for c in claims)), 'violations', len(bad), flush=True); tot += len(bad)
print('TOTAL violations', tot)
