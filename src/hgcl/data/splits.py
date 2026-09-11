"""Separate global targets and deterministic unique-address label budgets."""
import hashlib
import math
import polars as pl
from hgcl.config import digest


def budgets(wallets, targets, split, fractions, seeds):
    occurrences=wallets.select('address','step').unique().sort('address','step')
    development=set(occurrences.filter(pl.col('step').is_in(split['train']+split['validation']))['address'])
    train_seen=set(occurrences.filter(pl.col('step').is_in(split['train']))['address'])
    populations={}
    for partition,steps in split.items():
        pop=occurrences.filter(pl.col('step').is_in(steps)).join(targets,on='address',how='left')
        classes={c:sorted(pop.filter(pl.col('class')==c)['address'].unique().to_list()) for c in (1,2)}
        if not all(classes.values()):raise ValueError(f'{partition}: missing known class; recut/budget is invalid, no automatic expansion')
        populations[partition]=classes
    records=[]
    for seed in seeds:
        ranked={partition:{c:sorted(ids,key=lambda a:(hashlib.sha256(f'{seed}|{partition}|{c}|{a}'.encode()).digest(),a))
                           for c,ids in populations[partition].items()} for partition in ('train','validation')}
        for fraction in fractions:
            selected={};counts={}
            for partition in ('train','validation'):
                selected[partition]=[];counts[partition]={}
                for c in (1,2):
                    n=math.floor(fraction*len(ranked[partition][c]))
                    if n==0:raise ValueError(f'{partition}: zero class {c} at fraction {fraction}; no budget inflation')
                    selected[partition]+=ranked[partition][c][:n]
                    counts[partition][str(c)]={'available':len(ranked[partition][c]),'selected':n}
                selected[partition].sort()
            record={'seed':seed,'fraction':fraction,'selected':selected,'counts':counts,
                    'selected_overlap':len(set(selected['train']) & set(selected['validation']))}
            record['hash']=digest(record);records.append(record)
    recurrence=occurrences.with_columns(
        pl.col('address').is_in(sorted(train_seen)).alias('seen_in_train'),
        pl.col('address').is_in(sorted(development)).alias('seen_in_development'))
    return records,recurrence


def occurrence_masks(occurrences, targets, split, record):
    out=occurrences.select('address','step').unique().sort('address','step').join(targets,on='address',how='left')
    for partition in ('train','validation'):
        out=out.with_columns((pl.col('step').is_in(split[partition]) & pl.col('address').is_in(record['selected'][partition])).alias(partition+'_mask'))
        out=out.with_columns(pl.col('address').is_in(record['selected'][partition]).alias('label_exposed_'+partition))
    return out.with_columns((pl.col('step').is_in(split['test']) & pl.col('class').is_in([1,2])).alias('test_mask'),
                           pl.when(pl.col('class')==1).then(1).when(pl.col('class')==2).then(0).otherwise(None).cast(pl.Int8).alias('target'))
