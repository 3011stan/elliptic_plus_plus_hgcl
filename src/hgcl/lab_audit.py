"""Compact preparation evidence for operator review; no fitting or test metrics."""
from pathlib import Path
import json
import shutil

import polars as pl

from hgcl.pipeline import validate
from hgcl.provenance import source_identity


def audit(config, prepared):
    prepared = Path(prepared).resolve()
    source = source_identity(config.root)
    manifest = json.loads((prepared / 'manifest.json').read_text())
    if manifest['code_hash'] != source['sha256']:
        raise ValueError('Prepared source differs from current source; consult researcher')
    validation = validate(prepared, config)
    diagnostics = json.loads((prepared / 'diagnostics.json').read_text())
    records = json.loads((prepared / 'targets/budgets.json').read_text())
    regimes = {}
    for regime in config.values['features']['regimes']:
        snapshots = [dict(step=step, **diagnostics['snapshots'][f'{regime}/{step}'])
                     for step in sorted(sum(config.values['split'].values(), []))]
        regimes[regime] = {
            'snapshots': snapshots,
            'maxima': {key: max(row[key] for row in snapshots)
                       for key in ('addresses', 'transactions', 'edges')},
            'missingness': json.loads((prepared / regime / 'missingness.json').read_text()),
        }
    recurrence = pl.read_parquet(prepared / 'targets/recurrence.parquet')
    populations = {}
    for partition, steps in config.values['split'].items():
        frame = recurrence.filter(pl.col('step').is_in(steps))
        populations[partition] = {
            'observations': frame.height,
            'unique_addresses': frame['address'].n_unique(),
            'seen_in_train_observations': frame.filter(pl.col('seen_in_train')).height,
            'seen_in_development_observations': frame.filter(pl.col('seen_in_development')).height,
        }
    if source_identity(config.root)['sha256'] != source['sha256']:
        raise ValueError('Source changed during audit')
    return {
        'status': 'PASS', 'prepared': str(prepared), 'validation': validation,
        'source_sha256': source['sha256'], 'git_commit': source['git_commit'],
        'config_hash': config.hash, 'preparation_hash': manifest['preparation_hash'],
        'payload_hash': manifest['payload_hash'],
        'split': config.values['split'], 'source_counts': diagnostics['source_counts'],
        'selected_counts': diagnostics['selected_counts'], 'duplicates': diagnostics['duplicates'],
        'transaction_missingness': diagnostics['transaction_missingness'], 'regimes': regimes,
        'budgets': [{key: r[key] for key in ('seed', 'fraction', 'counts', 'selected_overlap', 'hash')}
                    for r in records],
        'recurrence': populations,
        'prepared_bytes': sum(p.stat().st_size for p in prepared.rglob('*') if p.is_file()),
        'disk_free_bytes': shutil.disk_usage(prepared).free,
        'training_performed': False, 'test_metrics_computed': False,
        'researcher_acceptance': 'pending',
        'notes': ['PASS certifies validation, not researcher acceptance.',
                  'Full validation checks global targets and masks; no performance selection.',
                  'Recurrence counts include unknown addresses; they are not evaluation support.',
                  'Preparation time and RSS remain in lab-prepare.json; GPU training cost is unmeasured.'],
    }
