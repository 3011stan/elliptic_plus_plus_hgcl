"""Preparation, training, immutable evaluation, matrix execution and reporting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from hgcl import __version__
from hgcl.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog='hgcl',description='S02 — H-GCL on Elliptic++')
    parser.add_argument('--version',action='version',version=__version__)
    subs=parser.add_subparsers(dest='command',required=True)
    for name in ('doctor','prepare','validate','fit','smoke','matrix'):
        sub=subs.add_parser(name)
        sub.add_argument('--config',type=Path,required=True)
        sub.add_argument('--data-root');sub.add_argument('--artifacts-root')
        sub.add_argument('--device',choices=['cpu','cuda']);sub.add_argument('--json',action='store_true')
        if name in ('validate','fit','smoke'):sub.add_argument('--prepared',type=Path,required=True)
        if name in ('fit','smoke'):sub.add_argument('--run-id',required=True)
        if name=='matrix':
            sub.add_argument('--prepared',type=Path)
            sub.add_argument('--matrix-id')
            sub.add_argument('--dry-run',action='store_true')
    for name in ('evaluate','resume'):
        sub=subs.add_parser(name);sub.add_argument('--run',type=Path,required=True)
    reporting=subs.add_parser('report');reporting.add_argument('--matrix',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command in ('evaluate','resume'):
            from hgcl.training.runner import evaluate,resume
            print(json.dumps((evaluate if args.command=='evaluate' else resume)(args.run),indent=2,allow_nan=False));return 0
        if args.command=='report':
            from hgcl.evaluation.reporting import report
            result=report(args.matrix)
            print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['status']=='complete' else 4
        config=load_config(args.config,data_root=args.data_root,artifacts_root=args.artifacts_root,device=args.device)
        if args.command=='doctor':
            if config.values['resources']['device']!='cpu':raise ValueError('CUDA doctor belongs to T030; currently CPU only')
            from hgcl.environment import doctor
            result=doctor(config.root)
        elif args.command=='matrix':
            from hgcl.matrix import dry_run,execute
            if args.dry_run:result=dry_run(config,args.prepared)
            else:
                if args.prepared is None or args.matrix_id is None:raise ValueError('Matrix execution requires --prepared and --matrix-id')
                result=execute(config,args.prepared,args.matrix_id)
        elif args.command=='prepare':
            from hgcl.pipeline import prepare
            result=prepare(config)
        elif args.command in ('fit','smoke'):
            from hgcl.training.runner import fit,smoke
            result=(fit if args.command=='fit' else smoke)(config,args.prepared,args.run_id)
        else:
            from hgcl.pipeline import validate
            result=validate(args.prepared,config)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,FileNotFoundError) as exc:
        print(json.dumps({'status':'error','error':str(exc)}),file=sys.stderr);return 4 if args.command=='resume' else 2
    except (RuntimeError,MemoryError,OSError) as exc:
        print(json.dumps({'status':'error','error':str(exc)}),file=sys.stderr);return 3


if __name__=='__main__':raise SystemExit(main())
