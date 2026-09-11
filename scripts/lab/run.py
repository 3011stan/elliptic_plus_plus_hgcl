#!/usr/bin/env python3
"""Operator entry point: one explicit stage, JSON receipt, no automatic next stage."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['download','doctor','smoke','prepare','matrix','report'])
    parser.add_argument('--id',help='New smoke run ID, or matrix ID for execution/restart/report')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if os.environ.get('HGCL_NIX_SYSTEM')!='x86_64-linux' or Path(sys.prefix).resolve()!=root/'.venv-lab':
        parser.error('Use nix develop, then .venv-lab/bin/python scripts/lab/run.py ...')
    from hgcl.config import load_config
    from hgcl.provenance import atomic_json,source_identity,file_hash
    config=load_config(root/'configs/lab.yaml')
    receipts=config.artifacts_root/'environment';receipts.mkdir(parents=True,exist_ok=True)
    output=receipts/('lab-'+args.stage+'.json')
    def require(stage):
        receipt=json.loads((receipts/('lab-'+stage+'.json')).read_text())
        if receipt.get('status') not in ('PASS','complete'):raise ValueError(f'{stage} has not passed')
        if receipt['source_sha256']!=source_identity(root)['sha256']:raise ValueError(f'{stage} belongs to different source/locks; repeat verification')
        return receipt
    try:
        if args.stage=='download':
            from hgcl.data.download import acquire
            result=acquire(config)
        elif args.stage=='doctor':
            from hgcl.environment import lab_doctor
            result=lab_doctor(config)
        elif args.stage=='smoke':
            require('doctor')
            if not args.id:raise ValueError('Smoke requires a new --id')
            from hgcl.pipeline import prepare
            from hgcl.training.runner import smoke
            smoke_config=load_config(root/'configs/smoke-gpu.yaml')
            prepared=prepare(smoke_config)
            result=smoke(smoke_config,Path(prepared['prepared']),args.id)
        elif args.stage=='prepare':
            require('doctor');require('smoke')
            from hgcl.pipeline import prepare
            result=prepare(config)
            result['status']='PASS'
        elif args.stage=='matrix':
            require('doctor');require('smoke');prepared=require('prepare')
            if not args.id:raise ValueError('Matrix requires --id')
            from hgcl.matrix import execute
            result=execute(config,Path(prepared['prepared']),args.id)
        else:
            if not args.id or Path(args.id).name!=args.id:raise ValueError('Report requires a matrix --id')
            from hgcl.evaluation.reporting import report
            result=report(config.artifacts_root/'matrices'/args.id)
        result['source_sha256']=source_identity(root)['sha256']
        atomic_json(output,result)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result.get('status') in ('PASS','complete') else 4
    except Exception as exc:
        atomic_json(output,{'status':'FAIL','stage':args.stage,'error':str(exc)})
        print(f'Parada: {exc}\nDiagnóstico: {output}\nTraga este resultado antes de continuar.',file=sys.stderr)
        return 3


if __name__=='__main__':raise SystemExit(main())
