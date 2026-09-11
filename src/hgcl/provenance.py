"""Content identities and atomic, explicit artifact lifecycle."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from hgcl.config import digest


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def source_identity(root: Path) -> dict:
    files = [p for folder in ('src', 'configs', 'tests', 'requirements') for p in (root / folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    files += [root / 'pyproject.toml', root / 'AGENTS.md']
    hashes = {str(p.relative_to(root)): file_hash(p) for p in sorted(files) if p.is_file()}
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'files': hashes, 'sha256': digest(hashes), 'git_commit': git('rev-parse', '--verify', 'HEAD'),
            'git_status': git('status', '--porcelain', '--untracked-files=all')}


TRANSITIONS = {'planned': {'validated', 'failed', 'interrupted'}, 'validated': {'fitting', 'complete', 'failed', 'interrupted'},
               'fitting': {'frozen', 'failed', 'interrupted'}, 'frozen': {'scored', 'failed', 'interrupted'},
               'scored': {'evaluated', 'failed', 'interrupted'}, 'evaluated': {'complete', 'failed'},
               'failed': set(), 'interrupted': set(), 'complete': set()}


def status(path: Path, state: str, **details):
    previous = json.loads(path.read_text()) if path.exists() else None
    if state not in TRANSITIONS or (previous is None and state != 'planned'):
        raise ValueError('Invalid initial state')
    if previous and state not in TRANSITIONS[previous['state']]:
        raise ValueError(f"Invalid transition {previous['state']} -> {state}")
    atomic_json(path, {'state': state, 'history': ([] if previous is None else previous['history'] + [previous['state']]), **details})
