"""Download only the nine originals, and never replace an existing source file."""
from pathlib import Path
import json
import os
import shutil
import tempfile
from hgcl.provenance import file_hash, atomic_json

DRIVE_FOLDER='1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l'


def matches(path, record):
    return path.is_file() and path.stat().st_size==record['bytes'] and file_hash(path)==record['sha256']


def acquire(config, client=None):
    expected=json.loads((config.root/config.values['provenance']['source_manifest']).read_text())['files']
    if len(expected)!=9 or any(Path(n).name!=n for n in expected):raise ValueError('Invalid nine-file source manifest')
    raw=config.data_root;raw.mkdir(parents=True,exist_ok=True)
    missing=[]
    for name,record in expected.items():
        path=raw/name
        if path.exists():
            if not matches(path,record):raise ValueError(f'Existing original differs: {name}; preserved for inspection')
        else:missing.append(name)
    staging=config.artifacts_root/'downloads';staging.mkdir(parents=True,exist_ok=True)
    if missing:
        if client is None:
            import gdown as client
        # Listing only: do not download notebooks, unrelated files, or remote paths.
        listing=client.download_folder(id=DRIVE_FOLDER,output=str(staging),quiet=True,
                                       use_cookies=False,skip_download=True)
        if listing is None:raise RuntimeError('Could not list public Drive folder')
        chosen={}
        for name in missing:
            found=[entry for entry in listing if Path(entry.path).name==name]
            if len(found)!=1:raise ValueError(f'Drive must contain exactly one {name}; found {len(found)}')
            chosen[name]=found[0].id
        required=sum(expected[n]['bytes'] for n in missing)
        if shutil.disk_usage(raw).free < required + max(expected[n]['bytes'] for n in missing):
            raise RuntimeError('Insufficient space for original files and verified copy')
        for name in missing:
            partial=staging/name
            if not matches(partial,expected[name]):
                if partial.exists():raise ValueError(f'Download checksum mismatch: {name}; preserved for inspection')
                output=client.download(id=chosen[name],output=str(partial),quiet=False,use_cookies=False,resume=True)
                if output is None:raise RuntimeError(f'Drive download failed: {name}')
            if not matches(partial,expected[name]):raise ValueError(f'Download checksum mismatch: {name}')
            fd,temporary=tempfile.mkstemp(prefix='.verified-',dir=raw);os.close(fd)
            temp=Path(temporary)
            try:
                shutil.copyfile(partial,temp)
                if not matches(temp,expected[name]):raise ValueError(f'Copied checksum mismatch: {name}')
                os.link(temp,raw/name)  # atomic publication; refuses overwrite even after a race
            finally:temp.unlink(missing_ok=True)
            partial.unlink()
    report={'status':'PASS','source_folder':DRIVE_FOLDER,'data_root':str(raw),
            'files':expected,'downloaded':missing,'scope':'Original sizes and SHA-256, not semantic validation'}
    atomic_json(staging/'verified.json',report)
    return report
