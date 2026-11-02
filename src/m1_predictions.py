"""M1: probabilities from the 10 released baseline checkpoints on the Validation and Test splits.

Saves results/phase1/preds/m1_seed<seed>_<split>.npz with file_id, probs (N x 9), y (0-8), domain (1-5),
partition (0 seen, 1 unseen, -1 not applicable) and results/phase1/m1_metrics.json (the Phase 0 gate numbers).
Runs with no DataLoader workers: evaluation is cheap, and worker processes re-import this module on Windows.
"""
import glob
import json
import os

import numpy as np

from common import CFG_PATH, REPO, RES, add_repo_to_path, ensure_dirs, log_to

add_repo_to_path()
from evaluate import evaluate_checkpoint  # noqa: E402  (official repo module)
from framework.config import resolve_config  # noqa: E402


def main():
    log = log_to('m1_predictions.log')
    with open(CFG_PATH, encoding='utf-8') as fh:
        config = resolve_config(json.load(fh))
    config['num_workers'] = 0
    ckpts = sorted(glob.glob(os.path.join(REPO, 'outputs', 'MTRCNN_seed*_B64_E100_earlystop_min10_pati5', 'model',
                                          'model_best.pth')))
    log('checkpoints found: %d' % len(ckpts))
    out_dir = os.path.join(RES, 'phase1', 'preds')
    ensure_dirs(out_dir)
    metrics_all = {}
    for ck in ckpts:
        seed = int(os.path.basename(os.path.dirname(os.path.dirname(ck))).split('seed')[1].split('_')[0])
        for split in ['validation', 'test']:
            out_npz = os.path.join(out_dir, 'm1_seed%d_%s.npz' % (seed, split))
            res = evaluate_checkpoint(config, ck, split, return_predictions=True)
            rows = res['predictions']
            probs = np.array([r['species_probabilities'] for r in rows], dtype=np.float32)
            y = np.array([r['true_species_index'] for r in rows], dtype=np.int64)
            dom = np.array([r['true_domain_index'] + 1 for r in rows], dtype=np.int64)
            part = np.array([{'seen': 0, 'unseen': 1}.get(r.get('evaluation_partition'), -1) for r in rows])
            fid = np.array([r['file_id'] for r in rows])
            np.savez_compressed(out_npz, file_id=fid, probs=probs, y=y, domain=dom, partition=part)
            m = res['metrics']
            key = 'seed%d_%s' % (seed, split)
            metrics_all[key] = {k: m.get(k) for k in ['species_accuracy', 'species_balanced_accuracy', 'BA_seen',
                                                      'BA_unseen', 'DSG', 'num_seen_samples', 'num_unseen_samples']}
            log('seed %d %s: %s' % (seed, split, json.dumps(metrics_all[key])))
    bu = [v['BA_unseen'] for k, v in metrics_all.items() if k.endswith('_test') and v['BA_unseen'] is not None]
    bs = [v['BA_seen'] for k, v in metrics_all.items() if k.endswith('_test') and v['BA_seen'] is not None]
    if bu:
        metrics_all['summary'] = {'BA_unseen_mean': float(np.mean(bu)), 'BA_unseen_std': float(np.std(bu)),
                                  'BA_seen_mean': float(np.mean(bs)), 'n_seeds': len(bu),
                                  'readme_reference': 'BA_unseen 0.185414 +- 0.013971, BA_seen 0.879283 (10 seeds)'}
        log('PHASE 0 GATE: ' + json.dumps(metrics_all['summary']))
    with open(os.path.join(RES, 'phase1', 'm1_metrics.json'), 'w', encoding='utf-8') as fh:
        json.dump(metrics_all, fh, indent=2)
    log('ALL DONE')


if __name__ == '__main__':
    main()
