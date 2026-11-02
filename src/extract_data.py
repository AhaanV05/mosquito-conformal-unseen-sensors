"""Extract the CD-MSC zips into data/cdmsc (leaves the originals untouched).

Usage: python extract_data.py [folder holding Development_data.zip and Evaluation_data.zip]
The folder defaults to ~/Downloads. Skips macOS junk (__MACOSX, .DS_Store).
Safe to re-run: files already extracted with the same size are skipped.
"""
import os
import sys
import time
import zipfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOADS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser('~'), 'Downloads')
OUT = os.path.join(PROJECT, 'data', 'cdmsc')
LOG = os.path.join(PROJECT, 'results', 'logs', 'extract.log')
os.makedirs(os.path.dirname(LOG), exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + msg
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


for zname, prefix in [('Development_data.zip', 'Development_data/'), ('Evaluation_data.zip', 'Evaluation_data/')]:
    zpath = os.path.join(DOWNLOADS, zname)
    if not os.path.exists(zpath):
        log('MISSING ' + zpath)
        continue
    with zipfile.ZipFile(zpath) as z:
        members = [i for i in z.infolist()
                   if i.filename.startswith(prefix) and not i.is_dir()
                   and '.DS_Store' not in i.filename and '__MACOSX' not in i.filename]
        log('%s: extracting %d files' % (zname, len(members)))
        t0 = time.time()
        done = skipped = 0
        for n, info in enumerate(members, 1):
            dst = os.path.join(OUT, info.filename.replace('/', os.sep))
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(info) as src, open(dst, 'wb') as out:
                out.write(src.read())
            done += 1
            if n % 20000 == 0:
                log('%s: %d/%d (%.0f s)' % (zname, n, len(members), time.time() - t0))
        log('%s: finished, wrote %d, skipped %d, %.0f s' % (zname, done, skipped, time.time() - t0))
log('ALL DONE')
