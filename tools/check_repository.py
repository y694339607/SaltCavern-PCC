# -*- coding: utf-8 -*-
"""Repository self-check: names referenced in code and docs must exist.

Checks
  1. every arm directory used to build `pred_<arm>` exists under predictions/
  2. every script that builds `pred_<m>` normalises `m` through an ALIAS map
  3. configuration files referenced anywhere exist in configs/
  4. scripts named in markdown exist in code/
  5. all Python files parse
Exit code is non-zero when any check fails.
"""
import ast
import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, 'predictions')
ARMS = {'adapointr_masked', 'adapointr_full', 'cylnn', 'spline', 'idw', 'pcn', 'pointr'}
LEGACY = {'BM', 'E2', 'PCN', 'PoinTr', 'B1', 'B2', 'B3'}

issues = []

present = set()
for base, dirs, _ in os.walk(PRED):
    for d in dirs:
        if d.startswith('pred_'):
            present.add(d[5:])

# 1 + 2
for f in glob.glob(os.path.join(ROOT, 'code', '**', '*.py'), recursive=True):
    t = io.open(f, encoding='utf-8', errors='replace').read()
    if 'f"pred_{m}' in t or "f'pred_{m}" in t:
        if 'ALIAS' not in t:
            issues.append('%s builds pred_{m} without an ALIAS map' % os.path.relpath(f, ROOT))
    for m in re.finditer(r'METHODS\s*=\s*\[ALIAS\.get\(m, m\) for m in (\[[^\]]*\])', t):
        toks = set(re.findall(r'"([^"]+)"', m.group(1)))
        if not toks <= LEGACY:
            issues.append('%s: unexpected METHODS tokens %s' % (os.path.relpath(f, ROOT), sorted(toks)))
        unknown = {a for a in ARMS if not os.path.isdir(os.path.join(PRED, 'test', 'pred_' + a))}
        if unknown:
            issues.append('arm directories missing under predictions/test: %s' % sorted(unknown))

# 3
cfgs = set(os.listdir(os.path.join(ROOT, 'configs')))
for f in glob.glob(os.path.join(ROOT, '**', '*'), recursive=True):
    if os.path.isfile(f) and f.endswith(('.py', '.md')):
        t = io.open(f, encoding='utf-8', errors='replace').read()
        for m in re.finditer(r'([A-Za-z0-9_\-]+\.yaml)', t):
            if m.group(1) not in cfgs:
                issues.append('%s references missing config %s' % (os.path.relpath(f, ROOT), m.group(1)))

# 4
scripts = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'code', '**', '*.py'), recursive=True)}
for f in glob.glob(os.path.join(ROOT, '**', '*.md'), recursive=True):
    t = io.open(f, encoding='utf-8', errors='replace').read()
    for m in re.finditer(r'`([A-Za-z0-9_\-]+\.py)`', t):
        if m.group(1) not in scripts:
            issues.append('%s mentions missing %s' % (os.path.relpath(f, ROOT), m.group(1)))

# 5
bad = []
for f in glob.glob(os.path.join(ROOT, 'code', '**', '*.py'), recursive=True):
    try:
        ast.parse(io.open(f, encoding='utf-8').read())
    except SyntaxError as e:
        bad.append('%s:%s' % (os.path.relpath(f, ROOT), e.lineno))

print('arm directories: %s' % sorted(present))
if bad:
    print('syntax errors: %s' % bad)
for i in sorted(set(issues)):
    print('ISSUE %s' % i)
print('checks failed: %d' % (len(set(issues)) + len(bad)))
sys.exit(1 if (issues or bad) else 0)
