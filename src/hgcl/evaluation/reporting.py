"""Explicit support, recurrence strata, paired differences and incomplete groups."""
from pathlib import Path
from collections import defaultdict
import json
import statistics
import polars as pl
from hgcl.provenance import atomic_json,file_hash
from hgcl.evaluation.metrics import metrics

METRICS=('f1_illicit','precision','recall','ap','mcc')


def strata(scored,threshold):
    groups={'overall':scored,'seen':scored.filter(pl.col('seen_in_development')),
            'unseen':scored.filter(~pl.col('seen_in_development')),
            'train_label_exposed':scored.filter(pl.col('label_exposed_train')),
            'validation_label_exposed':scored.filter(pl.col('label_exposed_validation')),
            'label_unexposed':scored.filter(~pl.col('label_exposed_train') & ~pl.col('label_exposed_validation'))}
    return {name:{**metrics(frame['target'].to_numpy(),frame['score'].to_numpy(),threshold),
                  'unique_addresses':frame['address'].n_unique()} for name,frame in groups.items()}


def summary(values,expected=5):
    finite=[v for v in values if v is not None]
    return {'n_available':len(values),'n_defined':len(finite),'expected_seeds':expected,
            'complete':len(values)==expected and len(finite)==expected,
            'mean':statistics.mean(finite) if finite else None,
            'sample_sd':statistics.stdev(finite) if len(finite)>1 else None}


def aggregate(rows,expected_keys):
    indexed={r['id']:r for r in rows}
    if len(indexed)!=len(rows):raise ValueError('Duplicate evaluation result keys')
    expected={r['id']:r for r in expected_keys}
    if set(indexed)-set(expected):raise ValueError('Unexpected evaluation result')
    grouped=defaultdict(list);paired=defaultdict(dict)
    for key,row in indexed.items():
        spec=expected[key]
        for field in ('regime','seed','fraction','method'):
            if row[field]!=spec[field]:raise ValueError('Evaluation metadata mismatch')
        for stratum,result in row['strata'].items():
            grouped[(row['regime'],row['fraction'],row['method'],stratum)].append((row['seed'],result))
            paired[(row['regime'],row['fraction'],stratum,row['seed'])][row['method']]=result
    aggregates=[]
    for (regime,fraction,method,stratum),records in sorted(grouped.items()):
        if len({seed for seed,_ in records})!=len(records):raise ValueError('Duplicate seed in aggregate')
        aggregates.append({'regime':regime,'fraction':fraction,'method':method,'stratum':stratum,
                           'seeds':sorted(seed for seed,_ in records),
                           'metrics':{m:summary([r[m] for _,r in records]) for m in METRICS},
                           'supports':[{'seed':seed,'observations':r['support'],'addresses':r['unique_addresses'],
                                        'positive':r['positive'],'negative':r['negative']} for seed,r in sorted(records)]})
    differences=defaultdict(list)
    comparisons=[('RQ1','fusion','rf'),('RQ1','fusion','hgcl'),('RQ1','fusion','graph_supervised'),('RQ2','hgcl','graph_supervised')]
    for (regime,fraction,stratum,seed),methods in paired.items():
        for rq,left,right in comparisons:
            if left not in methods or right not in methods:continue
            a,b=methods[left],methods[right]
            if (a['support'],a['positive'],a['negative'])!=(b['support'],b['positive'],b['negative']):raise ValueError('Paired populations have different support')
            value=None if a['f1_illicit'] is None or b['f1_illicit'] is None else a['f1_illicit']-b['f1_illicit']
            differences[(rq,regime,fraction,stratum,left,right)].append((seed,value))
    comparisons_out=[{'question':rq,'regime':regime,'fraction':fraction,'stratum':stratum,'difference':f'{left} minus {right}',
                      'paired_seeds':sorted(seed for seed,_ in values),'f1_difference':summary([v for _,v in values])}
                     for (rq,regime,fraction,stratum,left,right),values in sorted(differences.items())]
    missing=sorted(set(expected)-set(indexed))
    return {'status':'complete' if not missing else 'incomplete','expected_evaluations':len(expected),'available_evaluations':len(rows),
            'missing_evaluations':missing,'aggregates':aggregates,'paired_comparisons':comparisons_out,
            'limitations':['Five seeds are repetitions on one dataset, not independent datasets.',
                          'Native wallet comparison changes input policy; it is not a pure causal estimate of leakage.',
                          'Global labels do not establish when illicit activity began or was discovered.']}


def report(matrix):
    matrix=Path(matrix).resolve();manifest=json.loads((matrix/'manifest.json').read_text())
    states=json.loads((matrix/'groups.json').read_text()) if (matrix/'groups.json').exists() else {}
    rows=[]
    for expected in manifest['evaluations']:
        state=states.get(expected['group'],{})
        if state.get('state')!='complete':continue
        run=Path(state['run'])
        if file_hash(run/'metrics.json')!=state['metrics_hash']:raise ValueError('Report metrics identity mismatch')
        result=json.loads((run/'metrics.json').read_text())
        for field in ('regime','seed','fraction'):
            if result.get(field)!=expected[field]:raise ValueError('Reported run does not match matrix group')
        if expected['method'] not in result.get('strata',{}):continue
        rows.append({**expected,'strata':result['strata'][expected['method']]})
    output=aggregate(rows,manifest['evaluations']);atomic_json(matrix/'report.json',output)
    text=f"# S02 matrix report\n\nStatus: {output['status']}. Available evaluations: {len(rows)}/{len(manifest['evaluations'])}.\n\n"
    text+='Five-seed estimates and paired differences, including explicitly incomplete groups, are stored in report.json.\n'
    text+='Native wallet results describe an input-policy comparison. Global labels do not locate crime onset.\n'
    text+='\n| Regime | Labels | Method | Stratum | F1 mean | Sample SD | Defined seeds |\n| --- | --- | --- | --- | --- | --- | --- |\n'
    for row in output['aggregates']:
        m=row['metrics']['f1_illicit']
        text+=f"| {row['regime']} | {row['fraction']:.0%} | {row['method']} | {row['stratum']} | {m['mean']} | {m['sample_sd']} | {m['n_defined']}/5 |\n"
    text+='\nPaired differences are F1(left) − F1(right); negative values are retained. No significance claim is inferred.\n'
    (matrix/'report.md').write_text(text)
    return output
