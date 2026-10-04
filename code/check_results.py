"""Check the published result files (run from code/):  python check_results.py
 - every one of the 211,680 cells has a record with ok = true (results/cells.jsonl.gz)
 - sectors (results/sectors.zip): every final file has an empty failure list, and every file with failures has all
   its refinement chunks (the failed boxes are re-run in n = max(1, min(40, len(fail)//500 + 1)) chunks, tags ...c<j>)
 - tubes (results/tubes): every result file has an empty failure list; box and claim counts from the logs"""
import json, gzip, glob, os, zipfile, re, sys
# results directory: default ../results (the published files); pass another directory to check new runs
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
print('checking', os.path.abspath(R))
# expected inventory of the published proof (an empty or incomplete directory is a failure, not "ALL OK")
EXPECT = dict(sector_files=196, sector_final=120, sector_boxes=3613621, sector_starts=54,
              tubes={'tube_C_ref1': (6963, 1048), 'tube_C_ref2': (6963, 1048), 'tube_D_ref1': (6662, 1052)})
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
# every start region of every sector run (point, reference, mode, piece, face) must be present
starts = {re.match(r'sec_(\w+?_\w+?_\w+?_\d)_(\d)', n).group(1) + '_' + re.match(r'sec_(\w+?_\w+?_\w+?_\d)_(\d)', n).group(2) for n in files}
inv = []
if len(files) != EXPECT['sector_files'] or len(final) != EXPECT['sector_final']: inv.append('sector file count')
if ok != EXPECT['sector_boxes']: inv.append('sector box count')
if len(starts) != EXPECT['sector_starts']: inv.append('sector start regions %d' % len(starts))
print('sector start regions:', len(starts), '(expected', EXPECT['sector_starts'], ')')
tub = sorted(glob.glob(os.path.join(R, 'tubes', '*.json'))); tubbad = [f for f in tub if json.load(open(f))]
for name, (nbox, nclaim) in EXPECT['tubes'].items():
    lg = os.path.join(R, 'tubes', name + '_run.log')
    if not os.path.exists(lg): inv.append('missing ' + name); continue
    t = open(lg).read(); m = re.findall(r'RIGOROUS tube ok (\d+) fail (\d+)', t); c = re.findall(r'exact claims (\d+) violations (\d+)', t)
    print(' ', name, 'boxes ok/fail', m, 'exact claims/violations', c)
    if not m or m[-1] != (str(nbox), '0'): inv.append(name + ' boxes')
    if not c or any(x != (str(nclaim), '0') for x in c): inv.append(name + ' claims')
if len(tub) != len(EXPECT['tubes']): inv.append('tube result files %d' % len(tub))
print('tube result files:', len(tub), 'with failures', len(tubbad))
if inv: print('inventory problems:', inv)
okall = not (miss or notok or secbad or chainbad or tubbad or inv)
print('ALL OK' if okall else 'PROBLEM'); sys.exit(0 if okall else 1)
