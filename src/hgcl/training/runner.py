"""Fit/freeze/score/evaluate lifecycle for the original-data smoke milestone."""
from pathlib import Path
import json
import os
import re
import resource
import shutil
import sys
import time
import uuid
import joblib
import numpy as np
import polars as pl
import torch
import yaml
from hgcl.config import Config,digest
from hgcl.provenance import atomic_json,file_hash,source_identity,status
from hgcl.pipeline import validate
from hgcl.data.sampling import load_graph
from hgcl.models.heads import Classifier
from hgcl.models.tabular import fit_rf,rf_scores
from hgcl.training.pretrain import pretrain,fresh_encoder
from hgcl.training.supervised import fit_graph,graph_scores,permitted_targets
from hgcl.evaluation.selection import select_threshold,select_fusion
from hgcl.evaluation.metrics import metrics


class Guard:
    def __init__(self,resources,timed=True):
        self.resources=resources;self.start=time.monotonic();self.timed=timed;self.offset=0.
    @property
    def elapsed(self):return self.offset+time.monotonic()-self.start
    def __call__(self):
        if self.timed and self.resources['timeout_seconds'] is not None and self.elapsed>self.resources['timeout_seconds']:
            raise RuntimeError('Fitting and selection exceeded the configured time limit')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform=='darwin' else 1024**2)
        if rss>self.resources['rss_gib']:raise RuntimeError('RSS memory guard exceeded')
        if self.resources['device']=='cuda' and self.resources['gpu_gib'] is not None:
            if torch.cuda.memory_allocated()/1024**3>self.resources['gpu_gib']:raise RuntimeError('GPU memory guard exceeded')


def save_model(path,value,kind):
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    try:
        if kind=='torch':torch.save(value,temporary)
        else:joblib.dump(value,temporary)
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def aligned(scores):
    merged=None
    for name,frame in scores.items():
        frame=frame.sort('address','step')
        if frame.select('address','step').unique().height!=frame.height:raise ValueError('Duplicate prediction keys')
        if merged is None:merged=frame.rename({'score':name})
        else:
            if not merged.select('address','step').equals(frame.select('address','step')):raise ValueError('Prediction populations do not match')
            merged=merged.with_columns(frame['score'].alias(name))
    return merged


def fit(config,prepared,run_id,*,record=None,regime="principal",ssl_directory=None,resuming=False):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',run_id):raise ValueError('Invalid run ID')
    values=config.values
    if regime not in values['features']['regimes']:raise ValueError('Regime not in configuration')
    prepared=Path(prepared).resolve();validate(prepared,config,label_semantics=False)
    run=config.artifacts_root/'runs'/run_id
    if not resuming:
        run.mkdir(parents=True,exist_ok=False)
        status(run/'status.json','planned')
    else:
        if not run.is_dir():raise ValueError('Resume requires an existing run')
    stage='preflight';timer=None;attempt_started=not resuming
    try:
        manifest=json.loads((prepared/'manifest.json').read_text());records=json.loads((prepared/'targets/budgets.json').read_text())
        if record is None:record=next(r for r in records if r['seed']==values['labels']['seeds'][0] and r['fraction']==1.)
        if record not in records:raise ValueError('Label budget is not in verified preparation')
        if regime=='native_wallet' and record['fraction']!=1.:raise ValueError('Native complement requires 100% labels')
        source=source_identity(config.root)
        dependencies={'config':config.hash,'preparation':manifest['preparation_hash'],'payload':manifest['payload_hash'],
                      'source':source['sha256'],'mask':record['hash'],'regime':regime,'seed':record['seed'],
                      'environment_lock':file_hash(config.root/('requirements/mac-cpu.lock' if values['profile']=='smoke' else 'requirements/lab-cuda.lock'))}
        elapsed_offset=0.
        context={'regime':regime,'ssl_directory':str(Path(ssl_directory).resolve()) if ssl_directory else None}
        if resuming:
            if json.loads((run/'context.json').read_text())!=context:raise ValueError('SSL cache context changed')
            previous=json.loads((run/'provenance.json').read_text())
            if previous['dependencies']!=dependencies:raise ValueError('Source/data/config/mask changed; incompatible resume')
            previous_state=json.loads((run/'status.json').read_text())['state']
            if previous_state not in ('failed','interrupted','fitting','validated','planned') or (run/'predictions.parquet').exists() or (run/'frozen.json').exists():
                raise ValueError('Resume fitting forbidden after freeze or test scoring')
            attempt=run/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
            for name in ('status.json','failure.json','timings.json','provenance.json'):
                if (run/name).exists():shutil.copyfile(run/name,attempt/name)
            if (run/'failure.json').exists():elapsed_offset=json.loads((run/'failure.json').read_text()).get('elapsed_seconds') or 0.
            atomic_json(run/'status.json',{'state':'planned','history':['resume'],'previous_attempt':str(attempt)})
        attempt_started=True
        atomic_json(run/'context.json',context)
        atomic_json(run/'provenance.json',{'dependencies':dependencies,'prepared':str(prepared),'source':source,
                                         'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()})
        atomic_json(run/'config.json',values);(run/'resolved-config.yaml').write_text(yaml.safe_dump(values,sort_keys=False))
        atomic_json(run/'budget.json',record)
        for relative in source['files']:
            dest=run/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(config.root/relative,dest)
        status(run/'status.json','validated');status(run/'status.json','fitting')
        timer=Guard(values['resources']);timer.offset=elapsed_offset;timings={};reports={}
        stage='ssl';print('SSL pretraining...',file=sys.stderr,flush=True);started=time.monotonic()
        ssl_dependencies={k:v for k,v in dependencies.items() if k!='mask'}
        cache=Path(ssl_directory) if ssl_directory else run/'ssl'
        state,report=pretrain(prepared,regime,values,record['seed'],cache,ssl_dependencies,timer)
        if cache.resolve()!=(run/'ssl').resolve():
            (run/'ssl').mkdir(exist_ok=True)
            shutil.copyfile(cache/'ssl-final.pt',run/'ssl/ssl-final.pt')
        reports['ssl']=report;timings[stage]=time.monotonic()-started
        ssl_hash=file_hash(run/'ssl/ssl-final.pt')
        models={}
        for name,initial in [('graph_supervised',None),('hgcl',state)]:
            stage=name;print(f'Fitting {name}...',file=sys.stderr,flush=True);started=time.monotonic()
            models[name],reports[name]=fit_graph(prepared,regime,values,record,initial,run/name,{**dependencies,'method':name},timer)
            timings[name]=time.monotonic()-started
        if file_hash(run/'ssl/ssl-final.pt')!=ssl_hash:raise RuntimeError('Immutable SSL checkpoint changed during fine-tuning')
        stage='rf';print('Fitting Random Forest...',file=sys.stderr,flush=True);started=time.monotonic()
        models['rf'],reports['rf']=fit_rf(prepared,regime,values,record,run/'rf',timer);timings[stage]=time.monotonic()-started
        stage='selection';started=time.monotonic();scores={}
        for name in ('graph_supervised','hgcl'):
            scores[name]=graph_scores(models[name],prepared,regime,values['split']['validation'],values['resources'],timer)
        scores['rf']=rf_scores(models['rf'],prepared,regime,values['split']['validation'],timer)
        combined=aligned(scores);targets=permitted_targets(prepared,record,values,'validation')
        selected=targets.join(combined,on=['address','step'],how='left',validate='1:1');y=selected['target'].to_numpy()
        choices={name:select_threshold(y,selected[name].to_numpy()) for name in scores}
        choices['fusion'],search=select_fusion(y,selected['rf'].to_numpy(),selected['hgcl'].to_numpy(),values['fusion']['alphas'])
        pl.DataFrame(search).write_parquet(run/'validation-search.parquet');combined.write_parquet(run/'validation-predictions.parquet')
        atomic_json(run/'selection.json',choices)
        for name in ('graph_supervised','hgcl'):save_model(run/(name+'.pt'),{k:v.detach().cpu() for k,v in models[name].state_dict().items()},'torch')
        save_model(run/'rf.joblib',models['rf'],'joblib')
        reports['candidate_configurations']={name:len(reports[name]) for name in ('rf','graph_supervised','hgcl')}
        atomic_json(run/'training-report.json',reports)
        if source_identity(config.root)['sha256']!=source['sha256']:raise ValueError('Code changed during training')
        timings['selection']=time.monotonic()-started;timer()
        frozen_files=['context.json','config.json','budget.json','provenance.json','selection.json','graph_supervised.pt','hgcl.pt','rf.joblib','ssl/ssl-final.pt']
        frozen={'dependencies':dependencies,'files':{n:file_hash(run/n) for n in frozen_files}}
        atomic_json(run/'frozen.json',frozen);timer()
        timings['fitting_selection_seconds']=timer.elapsed
        atomic_json(run/'timings.json',timings);status(run/'status.json','frozen')
        return {'run':str(run),'status':'frozen','fitting_selection_seconds':timings['fitting_selection_seconds']}
    except BaseException as exc:
        if not attempt_started:raise
        atomic_json(run/'failure.json',{'stage':stage,'error':str(exc),'elapsed_seconds':timer.elapsed if timer else None})
        status(run/'status.json','interrupted' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',stage=stage,error=str(exc))
        raise


def load_models(run,prepared,values,record):
    regime=json.loads((run/'context.json').read_text())['regime'] if (run/'context.json').exists() else 'principal'
    graph=load_graph(prepared,regime,values['split']['train'][0]);models={}
    for name in ('graph_supervised','hgcl'):
        encoder=fresh_encoder(graph.x['address'].shape[1],graph.x['transaction'].shape[1],values,record['seed'])
        model=Classifier(encoder,values['encoder']['hidden'],record['seed'])
        model.load_state_dict(torch.load(run/(name+'.pt'),map_location='cpu',weights_only=True))
        models[name]=model.to(values['resources']['device']).eval()
    models['rf']=joblib.load(run/'rf.joblib')
    return models


def evaluate(run,*,verify_reload=False):
    run=Path(run).resolve();state=json.loads((run/'status.json').read_text())
    if state['state']!='frozen':raise ValueError('Evaluation requires a frozen, not yet evaluated run')
    started=time.monotonic()
    try:
        frozen=json.loads((run/'frozen.json').read_text())
        for name,sha in frozen['files'].items():
            if file_hash(run/name)!=sha:raise ValueError(f'Frozen artifact changed: {name}')
        values=json.loads((run/'config.json').read_text());record=json.loads((run/'budget.json').read_text())
        prepared=Path(json.loads((run/'provenance.json').read_text())['prepared'])
        manifest=json.loads((prepared/'manifest.json').read_text())
        if manifest['payload_hash']!=frozen['dependencies']['payload']:raise ValueError('Prepared data changed after freeze')
        # Verify integrity by hashes, without reading any target values before prediction.
        for name,sha in manifest['files'].items():
            if file_hash(prepared/name)!=sha:raise ValueError(f'Prepared artifact changed after freeze: {name}')
        regime=json.loads((run/'context.json').read_text())['regime'] if (run/'context.json').exists() else 'principal'
        choices=json.loads((run/'selection.json').read_text());guard=Guard(values['resources'],timed=False)
        def predict(models):
            scores={name:graph_scores(models[name],prepared,regime,values['split']['test'],values['resources'],guard) for name in ('graph_supervised','hgcl')}
            scores['rf']=rf_scores(models['rf'],prepared,regime,values['split']['test'],guard)
            combined=aligned(scores);alpha=choices['fusion']['alpha']
            return combined.with_columns((alpha*pl.col('rf')+(1-alpha)*pl.col('hgcl')).alias('fusion'))
        combined=predict(load_models(run,prepared,values,record))
        predictions=[]
        for name in values['evaluation']['methods']:
            frame=combined.select('address','step',pl.col(name).alias('score')).with_columns(pl.lit(name).alias('method'))
            frame=frame.with_columns((pl.col('score')>=choices[name]['threshold']).alias('prediction'))
            predictions.append(frame)
        predictions=pl.concat(predictions);predictions.write_parquet(run/'predictions.parquet')
        if verify_reload:
            reloaded=predict(load_models(run,prepared,values,record))
            for name in values['evaluation']['methods']:
                if not np.allclose(combined[name].to_numpy(),reloaded[name].to_numpy(),rtol=1e-7,atol=1e-8):raise RuntimeError('Reloaded predictions differ')
        status(run/'status.json','scored')
        # First materialization of test target rows happens after predictions are saved.
        test=(pl.scan_parquet(prepared/'targets'/(record['hash']+'.parquet'))
              .filter(pl.col('step').is_in(values['split']['test']) & pl.col('test_mask'))
              .select('address','step','target','label_exposed_train','label_exposed_validation').collect())
        recurrence=pl.read_parquet(prepared/'targets/recurrence.parquet')
        test=test.join(recurrence,on=['address','step'],how='left',validate='1:1')
        from hgcl.evaluation.reporting import strata
        output={};stratified={}
        for name in values['evaluation']['methods']:
            scored=test.join(combined.select('address','step',pl.col(name).alias('score')),on=['address','step'],how='left',validate='1:1')
            output[name]=metrics(scored['target'].to_numpy(),scored['score'].to_numpy(),choices[name]['threshold'])
            stratified[name]=strata(scored,choices[name]['threshold'])
        report={'scope':('engineering smoke, not scientific performance evidence' if values['profile']!='lab' else 'scientific temporal evaluation'), 'regime':regime,'seed':record['seed'],'fraction':record['fraction'],'methods':output,'strata':stratified,'reload_verified':verify_reload,
                'predictions_sha256':file_hash(run/'predictions.parquet'),'evaluation_seconds':time.monotonic()-started}
        atomic_json(run/'metrics.json',report);status(run/'status.json','evaluated');status(run/'status.json','complete')
        timings=json.loads((run/'timings.json').read_text());timings['final_evaluation_seconds']=report['evaluation_seconds'];atomic_json(run/'timings.json',timings)
        return {'run':str(run),'status':'complete','fitting_selection_seconds':timings['fitting_selection_seconds'],**report}
    except BaseException as exc:
        atomic_json(run/'evaluation-failure.json',{'error':str(exc)})
        status(run/'status.json','interrupted' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',error=str(exc))
        raise


def smoke(config,prepared,run_id):
    result=fit(config,prepared,run_id)
    return evaluate(result['run'],verify_reload=True)


def resume(run):
    run=Path(run).resolve()
    values=json.loads((run/'config.json').read_text())
    config=Config(json.dumps(values,sort_keys=True,allow_nan=False))
    saved=json.loads((run/'provenance.json').read_text())
    if source_identity(config.root)['sha256']!=saved['dependencies']['source']:
        raise ValueError('Source changed; incompatible resume')
    dep=saved['dependencies']
    if config.hash!=dep['config']:raise ValueError('Configuration changed; incompatible resume')
    prepared=Path(saved['prepared']);manifest=json.loads((prepared/'manifest.json').read_text())
    if manifest['preparation_hash']!=dep['preparation'] or manifest['payload_hash']!=dep['payload']:raise ValueError('Prepared data changed; incompatible resume')
    lock=config.root/('requirements/mac-cpu.lock' if values['profile']=='smoke' else 'requirements/lab-cuda.lock')
    if file_hash(lock)!=dep['environment_lock']:raise ValueError('Environment lock changed; incompatible resume')
    if json.loads((run/'budget.json').read_text())['hash']!=dep['mask']:raise ValueError('Label mask changed; incompatible resume')
    state=json.loads((run/'status.json').read_text())['state']
    if state=='frozen':return evaluate(run)
    if state in ('complete','scored','evaluated') or (run/'predictions.parquet').exists():
        raise ValueError('Cannot refit a test-scored/evaluated run')
    record=json.loads((run/'budget.json').read_text());context=json.loads((run/'context.json').read_text())
    result=fit(config,Path(saved['prepared']),run.name,record=record,regime=context['regime'],ssl_directory=context['ssl_directory'],resuming=True)
    return evaluate(result['run'])
