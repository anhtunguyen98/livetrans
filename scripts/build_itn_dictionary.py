"""Convert the VietNormalizer TTS lexicon to a conservative ITN whitelist.

Usage: python scripts/build_itn_dictionary.py --source /path/to/source.csv
Then: python -m app.custom_itn
"""
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
targets = defaultdict(set)
blocked_file = ROOT / 'assets/itn/blocked_aliases.txt'
blocked = {line.strip() for line in blocked_file.read_text().splitlines() if line.strip() and not line.startswith('#')} if blocked_file.exists() else set()
for row in csv.DictReader(args.source.read_text(encoding='utf-8-sig').splitlines()):
    written = unicodedata.normalize('NFC', row['original'].strip())
    spoken = unicodedata.normalize('NFC', row['transliteration'].strip().lower())
    spoken = re.sub(r'[-‐‑–]+', ' ', spoken)
    spoken = ' '.join(spoken.split())
    if written and spoken:
        targets[spoken].add(written)

accepted, review = {}, []
for spoken, originals in sorted(targets.items()):
    reason = None
    if spoken in blocked:
        reason = 'blocked: ordinary Vietnamese phrase'
    elif len(originals) != 1:
        reason = 'ambiguous pronunciation'
    elif len(spoken.split()) < 2:
        reason = 'single syllable; requires explicit approval/context'
    elif not all(re.fullmatch(r'[\w-]+', word, re.UNICODE) for word in originals):
        reason = 'unsupported written form'
    if reason:
        review.append({'spoken': spoken, 'written_candidates': sorted(originals), 'reason': reason})
    else:
        accepted[spoken] = next(iter(originals))

# Explicit overrides win over automatically reversed source entries.
overrides = ROOT / 'assets/itn/custom_whitelist.tsv'
for line in overrides.read_text().splitlines():
    if line.strip():
        written, spoken = line.split('\t')
        if spoken in blocked:
            raise ValueError(f'Explicit alias is blocked: {spoken}')
        accepted[spoken] = written
out = ROOT / 'assets/itn'
(out / 'generated_whitelist.tsv').write_text(
    ''.join(f'{written}\t{spoken}\n' for spoken, written in sorted(accepted.items())))
(out / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
manifest = {
    'source': 'https://github.com/nghimestudio/vietnormalizer/blob/main/public/non-vietnamese-words-20k.csv',
    'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
    'active_mappings': len(accepted), 'review_pronunciations': len(review),
    'note': 'Automatic candidates, not manually verified. Add ASR-specific overrides in custom_whitelist.tsv.',
}
(out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
