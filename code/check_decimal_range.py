import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_util
from decimal import Decimal as D
orig = dec_util.encl; stats = {'n':0, 'risky':0, 'minabove':None, 'maxbelow':D(0)}
def encl(x):
    a = abs(D(x)); stats['n'] += 1
    if a < D('1e-40'):
        if a > stats['maxbelow']: stats['maxbelow'] = a
    else:
        if stats['minabove'] is None or a < stats['minabove']: stats['minabove'] = a
    if D('1e-45') < a < D('1e-30'): stats['risky'] += 1
    return orig(x)
dec_util.encl = encl
import rig_sector; rig_sector.encl = encl
import rig_local2, rig_ball
from rig_sector import RigSector, EX
from rig_tube import RigTube
import rig_tube; rig_tube.encl = encl
from dec_util import mat
RZ = [[D(-1),D(0),D(0)],[D(0),D(-1),D(0)],[D(0),D(0),D(1)]]
permA = list(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'permA.npy')))
objs = [lambda: RigSector('A', mode='xi', r0=1.5e-3), lambda: RigSector('A', perm=permA, Mfix=RZ, mode='xi', r0=1.5e-3), lambda: RigSector('B', mode='xi', r0=1.5e-3)]
for w in 'CD':
    Tm = EX['T'+w]
    objs += [lambda Tm=Tm, w=w: RigSector(w, Vin_mat=Tm, mode='s', r0=1e-3),
             lambda Tm=Tm, w=w: RigSector(w, Vin_mat=Tm, mode='s', r0=1e-3, fan=np.array([0.0]), chifan=np.round(np.arange(-3,3.0001,0.05),10), chidir=0),
             lambda Tm=Tm, w=w: RigSector(w, Vin_mat=Tm, mode='s', r0=1e-3, fan=np.round(np.arange(-3,3.0001,0.025),10))]
objs += [lambda: RigTube('C', EX['TC'], 1, r0=1e-3, eps0=0.4, beta=0.5), lambda: RigTube('C', mat(EX['TC'], EX['h2C']), 1, Mfix=RZ, r0=1e-3, eps0=0.4, beta=0.5), lambda: RigTube('D', EX['TD'], -1, r0=1e-3, eps0=0.4, beta=0.5)]
for f in objs:
    f(); print(stats, flush=True)
print('RESULT: enclosed values', stats['n'], '; values with 1e-45 < |d| < 1e-30:', stats['risky'],
      '; largest |d| below 1e-40:', float(stats['maxbelow']), '; smallest |d| above 1e-40:', float(stats['minabove']))
