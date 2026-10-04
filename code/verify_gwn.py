"""Exact check of the norm bound |db| used in the sector certificates (rig_sector.py, line `gwn = ...`).
gwn is the square root of a floating-point sum of six outward-rounded squares; the sum itself is not rounded upward.
gwn depends only on the contact row (not on the box), so it is checked here once for every contact row of every sector
certificate used in the proof (the same constructions as rig_sector_run.py): the computed value must be at least the
exact square root of the exact (rational) sum of the six upper bounds.  Usage: python verify_gwn.py"""
import sys, numpy as np
from fractions import Fraction as F
sys.path.insert(0, '.')
from decimal import Decimal as D
from ia import up
from rig_sector import RigSector, EX
RZ = [[D(-1), D(0), D(0)], [D(0), D(-1), D(0)], [D(0), D(0), D(1)]]
permA = list(np.load('./permA.npy'))
def builds():
    yield 'A ref1', lambda **k: RigSector('A', mode='xi', r0=1.5e-3, **k), 'xi'
    yield 'A ref2', lambda **k: RigSector('A', perm=permA, Mfix=RZ, mode='xi', r0=1.5e-3, **k), 'xi'
    yield 'B ref1', lambda **k: RigSector('B', mode='xi', r0=1.5e-3, **k), 'xi'
    yield 'C ref1', lambda **k: RigSector('C', Vin_mat=EX['TC'], mode='s', r0=1e-3, **k), 's'
    yield 'D ref1', lambda **k: RigSector('D', Vin_mat=EX['TD'], mode='s', r0=1e-3, **k), 's'
tot = bad = 0
for name, mk, mode in builds():
    variants = [dict()]
    if mode == 's':
        variants += [dict(fan=np.array([0.0]), chifan=np.round(np.arange(-3, 3.0001, 0.05), 10), chidir=0),
                     dict(fan=np.round(np.arange(-3, 3.0001, 0.025), 10))]
    for kw in variants:
        R = mk(**kw); GW = R.GW
        sq = GW.abs().sqr(); hi = sq.hi.reshape(len(sq.hi), -1)
        fsum = hi.sum(1); g = up(np.sqrt(fsum))           # exactly as in rig_sector.py
        for j in range(len(hi)):
            ex = sum(F(float(x)) for x in hi[j])
            if F(float(g[j])) ** 2 < ex: bad += 1
        tot += len(hi); print(name, kw.keys() and list(kw.keys()), 'rows', len(hi), 'violations so far', bad, flush=True)
ok = bad == 0 and tot == 36974
print('rows checked', tot, 'violations', bad); print('OK' if ok else 'PROBLEM'); sys.exit(0 if ok else 1)
