"""Baseline log-mel features, memory-light version of the official extract_features.py.

- writes configs/cdmsc_local.json (paths into this project)
- training mean/std from a seeded random subsample of Training clips, streamed (no huge pickle in RAM)
- validation and test features with the official extract_split_features
The statistics file uses the official schema so the released checkpoints load it unchanged.
"""
import argparse
import contextlib
import json
import os
import random
import time

import numpy as np

from common import (CFG_PATH, FEAT, META, RAW, REPO, RES, add_repo_to_path, ensure_dirs, log_to, read_ids)

add_repo_to_path()
from extract_features import build_feature_extractor  # noqa: E402  (official repo module)
from framework.acoustic_feature import (extract_log_mel_feature, extract_split_features, save_feature_stats,  # noqa: E402
                                        split_feature_path)
from framework.config import config_signature, feature_signature_payload, resolve_config  # noqa: E402
from framework.utilization import choose_device  # noqa: E402

log = log_to('baseline_features.log')


def build_config():
    with open(os.path.join(REPO, 'configs', 'default_experiment.json'), encoding='utf-8') as fh:
        base = json.load(fh)
    base.update(dataset_root=RAW,
                train_ids_path=os.path.join(META, 'Training_ids.txt'),
                val_ids_path=os.path.join(META, 'Validation_ids.txt'),
                test_ids_path=os.path.join(META, 'Test_ids.txt'),
                feature_root=FEAT, output_root=os.path.join(RES, 'baseline_outputs'), num_workers=2)
    ensure_dirs(os.path.dirname(CFG_PATH), FEAT)
    with open(CFG_PATH, 'w', encoding='utf-8') as fh:
        json.dump(base, fh, indent=2)
    return resolve_config(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-stats', type=int, default=40000, help='training clips used for mean/std (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    config = build_config()
    device = choose_device(config['device'])
    log('device: %s' % device)
    extractor = build_feature_extractor(config, device)

    stats_file = os.path.join(FEAT, 'training_feature_stats.json')
    if not os.path.exists(stats_file):
        ids = read_ids('Training')
        rnd = random.Random(args.seed)
        sample = ids if args.n_stats <= 0 or args.n_stats >= len(ids) else rnd.sample(ids, args.n_stats)
        log('computing training stats on %d of %d training clips' % (len(sample), len(ids)))
        t0 = time.time()
        total = 0
        fsum = fsq = None
        for i, fid in enumerate(sample, 1):
            feat = extract_log_mel_feature(os.path.join(RAW, fid + '.wav'), extractor, config['sample_rate'],
                                           config['normalize_waveform'], device)
            if fsum is None:
                fsum = feat.sum(axis=0, dtype=np.float64)
                fsq = np.square(feat, dtype=np.float64).sum(axis=0, dtype=np.float64)
            else:
                fsum += feat.sum(axis=0, dtype=np.float64)
                fsq += np.square(feat, dtype=np.float64).sum(axis=0, dtype=np.float64)
            total += feat.shape[0]
            if i % 5000 == 0:
                log('stats %d/%d (%.0f s)' % (i, len(sample), time.time() - t0))
        mean = fsum / total
        std = np.sqrt(np.maximum(fsq / total - np.square(mean), 1e-12))
        payload = feature_signature_payload(config, 'training')
        stats = {'num_frames': int(total), 'feature_config_signature': config_signature(payload),
                 'feature_config_payload': payload, 'mean': mean.astype(np.float32).tolist(),
                 'std': std.astype(np.float32).tolist(), 'note': 'subsample of %d training clips' % len(sample)}
        save_feature_stats(stats, FEAT)
        log('stats done in %.0f s, frames=%d' % (time.time() - t0, total))
    else:
        log('stats file exists, skipping')

    for split in ['validation', 'test']:
        path = split_feature_path(FEAT, split)
        if os.path.exists(path):
            log('%s features exist, skipping' % split)
            continue
        log('extracting %s features' % split)
        t0 = time.time()
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            extract_split_features(config=config, split_name=split, extractor=extractor, device=device)
        log('%s done in %.0f s' % (split, time.time() - t0))
    log('ALL DONE')


if __name__ == '__main__':
    main()
