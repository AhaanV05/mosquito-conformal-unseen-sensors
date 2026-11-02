"""Shared paths and helpers for the CD-MSC experiments."""
import json
import os
import sys
import time

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(PROJECT, 'data', 'cdmsc', 'Development_data')
EVAL = os.path.join(PROJECT, 'data', 'cdmsc', 'Evaluation_data')
RAW = os.path.join(DEV, 'raw_audio')
META = os.path.join(DEV, 'metadata')
FEAT = os.path.join(PROJECT, 'data', 'cdmsc', 'features')
RES = os.path.join(PROJECT, 'results')
LOGS = os.path.join(RES, 'logs')
REPO = os.path.join(PROJECT, 'external', 'CD-MSC')
CFG_PATH = os.path.join(PROJECT, 'configs', 'cdmsc_local.json')

SPECIES = {1: 'Aedes aegypti', 2: 'Aedes albopictus', 3: 'Culex quinquefasciatus', 4: 'Anopheles gambiae',
           5: 'Anopheles arabiensis', 6: 'Anopheles dirus', 7: 'Culex pipiens', 8: 'Anopheles minimus',
           9: 'Anopheles stephensi'}
SP_SHORT = {1: 'Ae.aeg', 2: 'Ae.alb', 3: 'Cx.qui', 4: 'An.gam', 5: 'An.ara', 6: 'An.dir', 7: 'Cx.pip',
            8: 'An.min', 9: 'An.ste'}
NAME_TO_ID = {v: k for k, v in SPECIES.items()}
# taxonomy levels used later (index = species id - 1)
GENUS = {1: 'Aedes', 2: 'Aedes', 3: 'Culex', 4: 'Anopheles', 5: 'Anopheles', 6: 'Anopheles', 7: 'Culex',
         8: 'Anopheles', 9: 'Anopheles'}
# sibling group: An. gambiae and An. arabiensis belong to the same species complex
GROUP = {1: 'Ae.aeg', 2: 'Ae.alb', 3: 'Cx.qui', 4: 'An.gambiae-complex', 5: 'An.gambiae-complex', 6: 'An.dir',
         7: 'Cx.pip', 8: 'An.min', 9: 'An.ste'}


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def log_to(name):
    ensure_dirs(LOGS)
    path = os.path.join(LOGS, name)

    def log(msg):
        line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(msg)
        print(line, flush=True)
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')
    return log


def read_ids(split):
    """split in Training, Validation, Test, TrainVal."""
    with open(os.path.join(META, split + '_ids.txt'), encoding='utf-8') as fh:
        return [os.path.splitext(line.strip())[0] for line in fh if line.strip()]


def parse_id(fid):
    parts = fid.split('_')
    return int(parts[1]), int(parts[3])   # species id, domain id


def load_split_summary():
    with open(os.path.join(META, 'split_summary.json'), encoding='utf-8') as fh:
        return json.load(fh)


def unseen_domain_by_species_id():
    """Official definition: each species has one held-out (unseen) domain, e.g. {'Aedes aegypti': 'D3'}."""
    payload = load_split_summary()
    out = {}
    for name, dom in payload.get('unseen_domain_by_species', {}).items():
        out[NAME_TO_ID[name]] = int(str(dom).replace('D', ''))
    return out


def add_repo_to_path():
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
