"""Preparation only: no model fitting or performance-based data selection."""
from __future__ import annotations
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import uuid
import numpy as np
import polars as pl
from hgcl.config import Config,digest
from hgcl.provenance import atomic_json,file_hash,source_identity,status
from hgcl.data.ingest import verify_sources,load_inputs,load_targets
from hgcl.data.features import address_features,transaction_features
from hgcl.data.preprocessing import Preprocessor
from hgcl.data.snapshots import recut,make_snapshot
from hgcl.data.splits import budgets,occurrence_masks
from hgcl.data.schema import NATIVE_COLUMNS


def _memory_guard(config):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gib=rss/(1024**3) if sys.platform=='darwin' else rss/(1024**2)
    if gib>config.values['resources']['rss_gib']:raise RuntimeError(f'RSS guard exceeded: {gib:.2f} GiB')
    return gib


def prepare(config: Config) -> dict:
    start=time.monotonic();v=config.values
    folder=config.artifacts_root/'preparation-attempts'/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    status(folder/'status.json','planned')
    try:
        print('Verifying nine original source hashes...',file=sys.stderr,flush=True)
        sources=verify_sources(config.data_root,config.root/v['provenance']['source_manifest'])
        code=source_identity(config.root)
        atomic_json(folder/'worktree-provenance.json',code)
        # Working-tree status is attempt metadata, not deterministic input content.
        code.pop('git_status')
        key=digest({'schema':1,'config':v,'sources':sources,'code':code['sha256'],'git_commit':code['git_commit']})
        stage=folder/'prepared';stage.mkdir()
        atomic_json(stage/'resolved-config.json',v)
        atomic_json(stage/'source-identity.json',code)
        # An immutable copy, not merely a hash of a mutable worktree.
        for relative in code['files']:
            dest=stage/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(config.root/relative,dest)
        native='native_wallet' in v['features']['regimes']
        print('Reading canonical input columns and verifying snapshot endpoints...',file=sys.stderr,flush=True)
        inputs=load_inputs(config.data_root,native=native)
        _memory_guard(config)
        selected_steps=sorted(sum(v['split'].values(),[]))
        tx=recut(inputs.transactions,selected_steps,v['resources']['recut_transactions'],v['labels']['seeds'][0])
        edges=inputs.edges.join(tx.select('tx_id'),on='tx_id',how='semi')
        keys=edges.select('address','step').unique()
        wallets=inputs.wallets.join(keys,on=['address','step'],how='semi').sort('address','step')
        tx_features=transaction_features(tx,edges)
        tx_pre=Preprocessor.fit(tx_features.filter(pl.col('step').is_in(v['split']['train'])),kind='transaction')
        canonical=stage/'canonical';canonical.mkdir()
        tx_features.write_parquet(canonical/'transactions.parquet')
        wallets.write_parquet(canonical/'wallets.parquet')
        edges.write_parquet(canonical/'edges.parquet')
        atomic_json(stage/'transaction-preprocessor.json',tx_pre.to_dict())
        diagnostics={'source_counts':{'transactions':inputs.transactions.height,'wallet_pairs':inputs.wallets.height,'edges':inputs.edges.height},
                     'duplicates':inputs.diagnostics,'selected_counts':{'transactions':tx.height,'wallet_pairs':wallets.height,'edges':edges.height},
                     'transaction_missingness':{c:tx_features[c].null_count() for c in tx_pre.raw_names}, 'snapshots':{}}
        for regime in v['features']['regimes']:
            print(f'Preparing {regime} features and snapshots...',file=sys.stderr,flush=True)
            addresses=address_features(tx,wallets,edges) if regime=='principal' else wallets.select('address','step',*NATIVE_COLUMNS)
            pre=Preprocessor.fit(addresses.filter(pl.col('step').is_in(v['split']['train'])),kind=regime)
            target=stage/regime;target.mkdir()
            atomic_json(target/'address-preprocessor.json',pre.to_dict())
            addresses.write_parquet(target/'raw-address-features.parquet')
            atomic_json(target/'missingness.json',{c:addresses[c].null_count() for c in pre.raw_names})
            for step in selected_steps:
                a=addresses.filter(pl.col('step')==step);t=tx_features.filter(pl.col('step')==step);e=edges.filter(pl.col('step')==step)
                snapshot=make_snapshot(a,t,e,pre,tx_pre)
                path=target/f'step-{step:02}';path.mkdir()
                for name,array in snapshot.items():np.save(path/(name+'.npy'),array,allow_pickle=False)
                diagnostics['snapshots'][f'{regime}/{step}']={'addresses':a.height,'transactions':t.height,'edges':e.height,'address_dim':len(pre.names),'transaction_dim':len(tx_pre.names)}
                _memory_guard(config)
        # Only this stage opens global targets. Graph builders never receive them.
        print('Preparing separate label budgets and recurrence metadata...',file=sys.stderr,flush=True)
        targets=load_targets(config.data_root,inputs.wallets)
        label_dir=stage/'targets';label_dir.mkdir()
        targets.write_parquet(label_dir/'global.parquet')
        records,recurrence=budgets(wallets,targets,v['split'],v['labels']['fractions'],v['labels']['seeds'])
        recurrence.write_parquet(label_dir/'recurrence.parquet')
        atomic_json(label_dir/'budgets.json',records)
        for record in records:
            occurrence_masks(wallets,targets,v['split'],record).write_parquet(label_dir/(record['hash']+'.parquet'))
        atomic_json(stage/'diagnostics.json',diagnostics)
        # Check originals and code have not changed during this preparation.
        verify_sources(config.data_root,config.root/v['provenance']['source_manifest'])
        if source_identity(config.root)['sha256']!=code['sha256']:raise ValueError('Source code changed during preparation')
        hashes={str(p.relative_to(stage)):file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest={'schema':1,'preparation_hash':key,'config_hash':config.hash,'sources':sources,'files':hashes,
                  'code_hash':code['sha256'],'git_commit':code['git_commit'],'payload_hash':digest(hashes),'historical_evidence':{
                      name:file_hash(config.root/name) for name in ['docs/data/inv-001/report.json','docs/data/transaction-input-check.json']}}
        atomic_json(stage/'manifest.json',manifest)
        status(folder/'status.json','validated')
        destination=config.artifacts_root/'prepared'/key;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            existing=validate(destination,config)
            if existing['payload_hash']!=manifest['payload_hash']:raise ValueError('Repeat preparation content differs')
            # Retain independently rebuilt payloads for the reproducibility evidence.
        else:
            stage.rename(destination)
        result={'prepared':str(destination),'preparation_hash':key,'payload_hash':manifest['payload_hash'],
                'elapsed_seconds':time.monotonic()-start,'peak_rss_gib':_memory_guard(config),'attempt':str(folder)}
        atomic_json(folder/'timings.json',result);status(folder/'status.json','complete',result=result)
        return result
    except BaseException as exc:
        state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        status(folder/'status.json',state,error=str(exc),elapsed_seconds=time.monotonic()-start)
        raise


def validate(prepared: Path, config: Config, *, label_semantics: bool = True) -> dict:
    prepared=prepared.resolve();m=json.loads((prepared/'manifest.json').read_text())
    if m['schema']!=1 or m['config_hash']!=config.hash:raise ValueError('Prepared configuration mismatch')
    for relative,expected in m['files'].items():
        path=(prepared/relative).resolve()
        if prepared not in path.parents or not path.is_file() or file_hash(path)!=expected:
            raise ValueError(f'Prepared artifact hash mismatch: {relative}')
    if digest(m['files'])!=m['payload_hash']:raise ValueError('Invalid payload identity')
    for regime in config.values['features']['regimes']:
        for step in sorted(sum(config.values['split'].values(),[])):
            folder=prepared/regime/f'step-{step:02}'
            def arr(name):return np.load(folder/(name+'.npy'),allow_pickle=False)
            a=arr('address_x');t=arr('transaction_x')
            if a.shape!=(len(arr('address_ids')),123 if regime=='principal' else 110) or t.shape!=(len(arr('transaction_ids')),32):raise ValueError('Invalid snapshot dimension')
            if not np.isfinite(a).all() or not np.isfinite(t).all():raise ValueError('Nonfinite snapshot features')
            for role,limits in [('sender',(len(a),len(t))),('receiver',(len(t),len(a)))]:
                edge=arr(role)
                if edge.dtype!=np.int64 or edge.shape[0]!=2 or not np.array_equal(edge[::-1],arr(role+'_reverse')):raise ValueError('Invalid paired edges')
                if any(np.any(edge[i]<0) or np.any(edge[i]>=n) for i,n in enumerate(limits)):raise ValueError('Invalid edge index')
    # Validate labels independently after the graph-only checks above.
    from hgcl.data.ingest import require_references
    wallets=pl.read_parquet(prepared/'canonical/wallets.parquet')
    edges=pl.read_parquet(prepared/'canonical/edges.parquet')
    tx=pl.read_parquet(prepared/'canonical/transactions.parquet')
    require_references(edges,tx,['tx_id','step'],'saved transaction endpoints')
    require_references(edges,wallets,['address','step'],'saved address endpoints')
    require_references(wallets,edges,['address','step'],'saved wallet coverage')
    expected=[]
    if label_semantics:
        target=pl.read_parquet(prepared/'targets/global.parquet')
        expected,recurrence=budgets(wallets,target,config.values['split'],config.values['labels']['fractions'],config.values['labels']['seeds'])
        if expected!=json.loads((prepared/'targets/budgets.json').read_text()):raise ValueError('Saved label budgets differ from contract')
        if not recurrence.equals(pl.read_parquet(prepared/'targets/recurrence.parquet')):raise ValueError('Saved recurrence metadata differs')
        for record in expected:
            masks=occurrence_masks(wallets,target,config.values['split'],record)
            if not masks.equals(pl.read_parquet(prepared/'targets'/(record['hash']+'.parquet'))):raise ValueError('Saved masks differ from permitted label access')
    for regime in config.values['features']['regimes']:
        for step in sorted(sum(config.values['split'].values(),[])):
            folder=prepared/regime/f'step-{step:02}'
            for name,frame,column in [('address_ids',wallets,'address'),('transaction_ids',tx,'tx_id')]:
                observed=np.load(folder/(name+'.npy'),allow_pickle=False).tolist()
                wanted=frame.filter(pl.col('step')==step)[column].sort().to_list()
                if observed!=wanted:raise ValueError('Saved node identity mismatch')
    return {'status':'PASS','preparation_hash':m['preparation_hash'],'payload_hash':m['payload_hash'],
            'files_verified':len(m['files']),'budgets_verified':len(expected)}
