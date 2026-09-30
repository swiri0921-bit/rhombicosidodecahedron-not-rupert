"""Check the published result files (run from code/):  python check_results.py
 - every one of the 211,680 cells has a record with ok = true (results/cells.jsonl.gz)
 - sectors (results/sectors.zip): every final file has an empty failure list, and every file with failures has all
   its refinement chunks (the failed boxes are re-run in n = max(1, min(40, len(fail)//500 + 1)) chunks, tags ...c<j>)
 - tubes (results/tubes): every result file has an empty failure list; box and claim counts from the logs"""
import json, gzip, glob, os, zipfile, re
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
cells = {}
with gzip.open(os.path.join(R, 'cells.jsonl.gz'), 'rt') as f:
    for l in f:
        d = json.loads(l); cells[d['k']] = d['ok']
miss = set(range(211680)) - set(cells); notok = [k for k, v in cells.items() if not v]
print('cells:', len(cells), 'missing', len(miss), 'not ok', len(notok))
Z = zipfile.ZipFile(os.path.join(R, 'sectors.zip'))
files = {os.path.basename(n)[:-5]: json.loads(Z.read(n)) for n in Z.namelist() if n.endswith('.json')}
final = [n for n in Z.namelist() if n.startswith('sectors/final/') and n.endswith('.json')]
secbad = [n for n in final if json.loads(Z.read(n))['fail']]
chainbad = []
for name, d in files.items():
    if d['fail']:
        n = max(1, min(40, len(d['fail']) // 500 + 1))
        for j in range(n):
            if name + f'c{j}' not in files: chainbad.append(name + f'c{j}')
ok = sum(d['ok'] for d in files.values())
print('sector result files:', len(files), '(final', len(final), ') final with failures', len(secbad),
      '; missing refinement chunks', len(chainbad), '; certified boxes', ok)
tub = sorted(glob.glob(os.path.join(R, 'tubes', '*.json'))); tubbad = [f for f in tub if json.load(open(f))]
for lg in sorted(glob.glob(os.path.join(R, 'tubes', '*.log'))):
    t = open(lg).read(); m = re.findall(r'RIGOROUS tube ok (\d+) fail (\d+)', t); c = re.findall(r'exact claims (\d+) violations (\d+)', t)
    print(' ', os.path.basename(lg), 'boxes ok/fail', m, 'exact claims/violations', c)
print('tube result files:', len(tub), 'with failures', len(tubbad))
print('ALL OK' if not (miss or notok or secbad or chainbad or tubbad) else 'PROBLEM')
