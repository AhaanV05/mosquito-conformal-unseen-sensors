"""Phase 0 audit of the CD-MSC development data (CPU only, read-only).

Writes results/phase0/phase0_audit.md and CSV tables.
"""
import json
import os
import random

import pandas as pd
import soundfile as sf

from common import (META, PROJECT, RAW, RES, SP_SHORT, SPECIES, ensure_dirs, load_split_summary, log_to, parse_id,
                    read_ids, unseen_domain_by_species_id)

log = log_to('phase0_audit.log')
OUT = os.path.join(RES, 'phase0')
ensure_dirs(OUT)

splits = {n: read_ids(n) for n in ['Training', 'Validation', 'Test', 'TrainVal']}
log('split sizes: ' + json.dumps({k: len(v) for k, v in splits.items()}))
rows = []
for name in ['Training', 'Validation', 'Test']:
    for fid in splits[name]:
        s, d = parse_id(fid)
        rows.append((fid, s, d, name))
df = pd.DataFrame(rows, columns=['id', 'species', 'domain', 'split'])
missing = [fid for fid in df['id'] if not os.path.exists(os.path.join(RAW, fid + '.wav'))]
log('ids missing on disk: %d of %d' % (len(missing), len(df)))
n_wav = len([x for x in os.listdir(RAW) if x.endswith('.wav')])
log('wav files in raw_audio: %d' % n_wav)

unseen = unseen_domain_by_species_id()
log('official unseen domain by species: ' + json.dumps({SP_SHORT[k]: 'D%d' % v for k, v in unseen.items()}))
df['unseen_official'] = [unseen.get(s) == d for s, d in zip(df['species'], df['domain'])]
tv = df[df['split'].isin(['Training', 'Validation'])]
pairs = set(zip(tv['species'], tv['domain']))
df['pair_seen_in_trainval'] = [(s, d) in pairs for s, d in zip(df['species'], df['domain'])]

tab_all = pd.crosstab(df['species'].map(SP_SHORT), 'D' + df['domain'].astype(str), margins=True)
tab_train = pd.crosstab(df[df['split'] == 'Training']['species'].map(SP_SHORT),
                        'D' + df[df['split'] == 'Training']['domain'].astype(str), margins=True)
te = df[df['split'] == 'Test']
tab_test_part = pd.crosstab(te['species'].map(SP_SHORT), te['unseen_official'].map({True: 'unseen', False: 'seen'}),
                            margins=True)
tab_test_dom = pd.crosstab(te['species'].map(SP_SHORT), 'D' + te['domain'].astype(str), margins=True)
agree = (te['unseen_official'] == (~te['pair_seen_in_trainval'])).mean()
log('agreement between official unseen flag and "pair absent from trainval" on test clips: %.4f' % agree)
unseen_in_train = df[(df['split'].isin(['Training', 'Validation'])) & (df['unseen_official'])]
log('official-unseen clips found in Training/Validation: %d' % len(unseen_in_train))

random.seed(0)
arows = []
for d, g in df.groupby('domain'):
    pick = g.sample(min(150, len(g)), random_state=0)
    for fid in pick['id']:
        info = sf.info(os.path.join(RAW, fid + '.wav'))
        arows.append((d, info.samplerate, info.duration, info.channels))
au = pd.DataFrame(arows, columns=['domain', 'sr', 'dur', 'ch'])
audio_tab = au.groupby('domain').agg(
    n=('sr', 'size'), sample_rates=('sr', lambda s: sorted(set(int(x) for x in s))),
    dur_mean=('dur', 'mean'), dur_min=('dur', 'min'), dur_max=('dur', 'max'),
    channels=('ch', lambda s: sorted(set(int(x) for x in s)))).round(2)

tab_all.to_csv(os.path.join(OUT, 'species_by_domain_all.csv'))
tab_train.to_csv(os.path.join(OUT, 'species_by_domain_training.csv'))
tab_test_part.to_csv(os.path.join(OUT, 'test_seen_unseen_by_species.csv'))
tab_test_dom.to_csv(os.path.join(OUT, 'test_species_by_domain.csv'))
audio_tab.to_csv(os.path.join(OUT, 'audio_by_domain.csv'))
df.to_csv(os.path.join(OUT, 'all_clips_with_split.csv'), index=False)

md = ['# Phase 0 audit (CD-MSC development data)', '',
      '- wav files on disk: **%d** (README says 271,380 plus the dataset has %d ids across splits)' % (n_wav, len(df)),
      '- ids missing on disk: **%d**' % len(missing),
      '- split sizes: ' + json.dumps({k: len(v) for k, v in splits.items()}),
      '- official unseen domain per species: ' + json.dumps({SP_SHORT[k]: 'D%d' % v for k, v in unseen.items()}),
      '- agreement of official unseen flag with "(species, domain) pair absent from Training+Validation" on test clips: **%.4f**' % agree,
      '- official-unseen clips found in Training/Validation: **%d** (should be 0)' % len(unseen_in_train), '',
      '## Clips per species x domain (all splits)', '', tab_all.to_markdown(), '',
      '## Training clips per species x domain', '', tab_train.to_markdown(), '',
      '## Test clips: seen vs unseen (official definition)', '', tab_test_part.to_markdown(), '',
      '## Test clips per species x domain', '', tab_test_dom.to_markdown(), '',
      '## Audio properties per domain (sample of up to 150 clips each)', '', audio_tab.to_markdown(), '',
      '## split_summary.json (first 3000 characters)', '', '```json',
      json.dumps(load_split_summary(), indent=2)[:3000], '```', '']
with open(os.path.join(OUT, 'phase0_audit.md'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(md))
log('wrote ' + os.path.join(OUT, 'phase0_audit.md'))
