"""Overnight orchestrator (resumable): runs the remaining stages one at a time and logs everything.

Order: baseline features (skipped if present) -> M1 predictions -> wait for M2 -> Phase 1 analysis on M1+M2
-> M3 (best effort, time-boxed) -> final Phase 1 analysis with every family that exists.
A failing step is logged and skipped so later independent steps still run. Run from the project root with the venv python.
"""
import json
import os
import shutil
import subprocess
import sys
import time

from common import FEAT, LOGS, PROJECT, RES, ensure_dirs, log_to

log = log_to('pipeline.log')
PY = sys.executable
PRED = os.path.join(RES, 'phase1', 'preds')
P1 = os.path.join(RES, 'phase1')


def run(name, cmd, timeout_min=None):
    out_path = os.path.join(LOGS, name + '.out')
    t0 = time.time()
    log('START %s: %s' % (name, ' '.join(cmd)))
    try:
        with open(out_path, 'w', encoding='utf-8') as fh:
            p = subprocess.run(cmd, cwd=PROJECT, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout_min * 60 if timeout_min else None)
        log('END %s: exit %d in %.1f min' % (name, p.returncode, (time.time() - t0) / 60))
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        log('TIMEOUT %s after %s min (killed)' % (name, timeout_min))
        return False
    except Exception as e:  # noqa: BLE001
        log('ERROR %s: %r' % (name, e))
        return False


def wait_for(check, what, timeout_min, poll=30):
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        if check():
            return True
        time.sleep(poll)
    log('gave up waiting for %s after %d min' % (what, timeout_min))
    return False


ensure_dirs(LOGS)
log('pipeline (resume mode) started; python %s' % PY)

# 1. baseline features
need = ['training_feature_stats.json', 'validation_features.pkl', 'test_features.pkl']
if all(os.path.exists(os.path.join(FEAT, f)) for f in need):
    log('baseline features already present, skipping')
    ok_feat = True
else:
    ok_feat = run('baseline_features', [PY, 'src/baseline_features.py', '--n-stats', '40000'], timeout_min=150)

# 2. M1 predictions (the metrics file is written last, so its presence means the stage finished)
m1_metrics = os.path.join(P1, 'm1_metrics.json')
if os.path.exists(m1_metrics):
    ok_m1 = True
    log('M1 predictions already present, skipping')
else:
    ok_m1 = run('m1_predictions', [PY, 'src/m1_predictions.py'], timeout_min=120) if ok_feat else False
if ok_m1:
    try:
        with open(m1_metrics, encoding='utf-8') as fh:
            log('PHASE 0 GATE (baseline reproduction): ' + json.dumps(json.load(fh).get('summary')))
    except Exception as e:  # noqa: BLE001
        log('could not read m1 metrics: %r' % (e,))

# 3. M2 (already produced); wait briefly in case it is still writing
wait_for(lambda: os.path.exists(os.path.join(PRED, 'm2_seed2_test.npz')), 'M2 predictions', 30)

# 4. main Phase 1 analysis on M1 + M2 (no dependency on the optional M3)
fams = ','.join(f for f, ok in [('m1', ok_m1), ('m2', True)] if ok)
if run('phase1_conformal_m1m2', [PY, 'src/phase1_conformal.py', '--families', fams], timeout_min=60):
    for name in ['phase1_report.md']:
        src = os.path.join(P1, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(P1, 'phase1_report_m1_m2.md'))

# 5. M3 best effort, time-boxed
run('m3_ast_probe', [PY, 'src/m3_ast_probe.py', '--workers', '0', '--batch', '16', '--cap', '1500'], timeout_min=150)

# 6. final analysis with every family that exists
if os.path.exists(os.path.join(PRED, 'm3_seed0_test.npz')):
    run('phase1_conformal_all', [PY, 'src/phase1_conformal.py', '--families', fams + ',m3'], timeout_min=60)
else:
    log('M3 predictions missing; the M1+M2 report stays the final one')
log('PIPELINE FINISHED')
