"""Bounded, label-free check of proposed named transaction inputs on train steps only."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['total_BTC', 'fees', 'size', 'num_input_addresses', 'num_output_addresses'] + [
    f'{direction}_BTC_{stat}' for direction in ('in', 'out')
    for stat in ('min', 'max', 'mean', 'median', 'total')]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    path = ROOT / 'elliptic-plus-plus/raw/txs_features.csv'
    manifest = json.loads((ROOT / 'docs/data/source-manifest.json').read_text())
    before = sha(path)
    if before != manifest['files'][path.name]['sha256']:
        raise ValueError('Original transaction file differs from manifest')
    stats = {k: {'finite': 0, 'missing': 0, 'negative': 0, 'zero': 0,
                 'nonfinite': 0, 'nonnumeric': 0, 'min': None, 'max': None} for k in FIELDS}
    counts = Counter()
    with path.open(newline='') as f:
        rows = csv.DictReader(f)
        if not set(FIELDS + ['txId', 'Time step']) <= set(rows.fieldnames):
            raise ValueError('Required column missing')
        for row in rows:
            if not 1 <= int(row['Time step']) <= 28:
                continue
            counts['training_transactions'] += 1
            vals = {}
            for k in FIELDS:
                s, raw = stats[k], row[k]
                if raw.strip() == '':
                    s['missing'] += 1
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    s['nonnumeric'] += 1
                    continue
                if not math.isfinite(v):
                    s['nonfinite'] += 1
                    continue
                vals[k] = v
                s['finite'] += 1
                s['negative'] += v < 0
                s['zero'] += v == 0
                s['min'] = v if s['min'] is None else min(s['min'], v)
                s['max'] = v if s['max'] is None else max(s['max'], v)
            if len(vals) == len(FIELDS):
                counts['all_15_finite'] += 1
                if all(v == 0 for v in vals.values()):
                    counts['all_15_zero'] += 1
                if abs(vals['in_BTC_total'] - vals['out_BTC_total'] - vals['fees']) > 1e-7:
                    counts['input_output_fee_difference_gt_1e-7_BTC'] += 1
                if abs(vals['total_BTC'] - vals['out_BTC_total']) > 1e-7:
                    counts['total_BTC_differs_from_output_total_gt_1e-7_BTC'] += 1
    if sha(path) != before:
        raise ValueError('Input changed during check')
    report = {'scope': 'Numerical diagnostics on steps 1–28 only; no labels read, no training.',
              'source': path.name, 'source_sha256': before,
              'script_sha256': sha(Path(__file__)), 'columns': FIELDS,
              'counts': dict(counts), 'statistics': stats,
              'limitations': ['Does not validate extraction code, size unit or historical label availability.',
                             'Financial identities are diagnostics, not automatic row exclusion rules.',
                             'All-zero vectors are ambiguous, not established missing-value sentinels.']}
    out = ROOT / 'docs/data/transaction-input-check.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(out), 'counts': dict(counts),
                      'issues': {k: {a: v for a, v in s.items() if a in
                          ('missing', 'negative', 'nonfinite', 'nonnumeric') and v}
                          for k, s in stats.items() if any(s[a] for a in
                          ('missing', 'negative', 'nonfinite', 'nonnumeric'))}}, indent=2))


if __name__ == '__main__':
    main()
