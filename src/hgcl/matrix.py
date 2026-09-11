"""Accepted 80+20 evaluation matrix; serial execution with ten shared SSL caches."""
from pathlib import Path
import json
import re
from hgcl.provenance import atomic_json,file_hash,source_identity
from hgcl.pipeline import validate
from hgcl.training.runner import fit,evaluate,resume

METHODS=('rf','graph_supervised','hgcl','fusion')


def design(values):
    if values['profile']!='lab':raise ValueError('The 100-evaluation matrix requires the lab profile')
    groups=[];evaluations=[];nodes=[]
    for regime in ('principal','native_wallet'):
        for seed in values['labels']['seeds']:
            ssl=f'ssl-{regime}-{seed}';nodes.append({'id':ssl,'kind':'ssl','depends_on':[]})
            for fraction in (values['labels']['fractions'] if regime=='principal' else [1.]):
                key=f'{regime}-{seed}-{int(fraction*100):03}'
                groups.append({'id':key,'regime':regime,'seed':seed,'fraction':fraction,'ssl':ssl})
                for method in ('rf','graph_supervised','hgcl'):
                    nodes.append({'id':f'fit-{key}-{method}','kind':'fit','depends_on':[ssl] if method=='hgcl' else []})
                for method in METHODS:
                    evaluation={'id':f'{key}-{method}','group':key,'regime':regime,'seed':seed,'fraction':fraction,'method':method}
                    evaluations.append(evaluation)
                    deps=[f'fit-{key}-{m}' for m in (('rf','hgcl') if method=='fusion' else (method,))]
                    nodes.append({'id':evaluation['id'],'kind':'evaluation','depends_on':deps})
    if len(evaluations)!=100 or len({e['id'] for e in evaluations})!=100 or len(groups)!=25:
        raise ValueError('Matrix must have exactly 100 unique evaluation keys in 25 groups')
    ids={n['id'] for n in nodes}
    if any(d not in ids for n in nodes for d in n['depends_on']):raise ValueError('Unresolved DAG dependency')
    return {'groups':groups,'evaluations':evaluations,'nodes':nodes,'expected_evaluations':100,'expected_ssl_caches':10,
            'expected_candidate_configurations':len(groups)*(2*len(values['supervised']['lrs'])+len(values['rf']['min_leaf']))}


def dry_run(config,prepared=None):
    result=design(config.values)
    integrity=validate(Path(prepared),config,label_semantics=False) if prepared is not None else None
    return {**result,'status':'planned','data_integrity_checked':integrity is not None,
            'preparation':integrity,'training_performed':False,'test_labels_materialized':False}


def execute(config,prepared,matrix_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,39}',matrix_id):raise ValueError('Invalid matrix ID')
    plan=dry_run(config,prepared);prepared=Path(prepared).resolve()
    source=source_identity(config.root);data=json.loads((prepared/'manifest.json').read_text())
    identity={'config':config.hash,'source':source['sha256'],'prepared':data['preparation_hash'],'payload':data['payload_hash'],
              'environment_lock':file_hash(config.root/'requirements/lab-cuda.lock')}
    folder=config.artifacts_root/'matrices'/matrix_id;folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['identity']!=identity:raise ValueError('Incompatible matrix restart: source/data/config/lock changed')
        if manifest['evaluations']!=plan['evaluations']:raise ValueError('Matrix evaluation keys changed')
    else:
        manifest={**plan,'identity':identity,'config':config.values,'prepared_path':str(prepared),'matrix_id':matrix_id}
        atomic_json(manifest_path,manifest)
    records=json.loads((prepared/'targets/budgets.json').read_text())
    states_path=folder/'groups.json';states=json.loads(states_path.read_text()) if states_path.exists() else {}
    for group in plan['groups']:
        record=next((r for r in records if r['seed']==group['seed'] and r['fraction']==group['fraction']),None)
        if record is None:raise ValueError(f"Missing verified budget: {group['id']}")
        run_id=f"{matrix_id}-{group['id']}";run=config.artifacts_root/'runs'/run_id
        try:
            if run.exists():
                dependencies=json.loads((run/'provenance.json').read_text())['dependencies']
                expected={'config':identity['config'],'source':identity['source'],'preparation':identity['prepared'],
                          'payload':identity['payload'],'environment_lock':identity['environment_lock'],
                          'mask':record['hash'],'regime':group['regime'],'seed':group['seed']}
                if dependencies!=expected:raise ValueError('Run dependencies do not match matrix group')
                current=json.loads((run/'status.json').read_text())['state']
                if current=='complete':
                    saved=states.get(group['id'],{})
                    if saved.get('metrics_hash') and saved['metrics_hash']!=file_hash(run/'metrics.json'):raise ValueError('Completed metrics changed')
                    result={'status':'complete','run':str(run)}
                else:result=resume(run)
            else:
                fitted=fit(config,prepared,run_id,record=record,regime=group['regime'],ssl_directory=folder/'ssl'/group['ssl'])
                result=evaluate(fitted['run'])
            states[group['id']]={'state':result['status'],'run':str(run),'metrics_hash':file_hash(run/'metrics.json')}
            atomic_json(states_path,states)
        except BaseException as exc:
            states[group['id']]={'state':'incomplete','run':str(run),'reason':str(exc)}
            atomic_json(states_path,states)
            raise
    return {'matrix':str(folder),'status':'complete','completed_groups':len(states),'evaluations':100}
