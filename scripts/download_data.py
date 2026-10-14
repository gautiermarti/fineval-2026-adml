#!/usr/bin/env python3
"""Download or import the exact benchmark files recorded in data/manifest.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def destination(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to((root/'data').resolve()) or not relative.startswith('data/'):
        raise ValueError(f'Invalid dataset path: {relative}')
    return path


def install(entry: dict, root: Path, source_root: Path | None = None, verify_only: bool = False):
    target = destination(root, entry['path'])
    expected = entry['sha256']
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f'Existing file has a different hash: {entry["path"]}; preserve it separately before retrying')
        return 'verified'
    if verify_only:
        raise FileNotFoundError(entry['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    try:
        if source_root is not None:
            source = destination(source_root, entry['path'])
            if sha256(source) != expected:
                raise ValueError(f'Local source hash mismatch: {entry["path"]}')
            shutil.copyfile(source, temporary)
        else:
            with urlopen(entry['url'], timeout=60) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
        if sha256(temporary) != expected:
            raise ValueError(f'Downloaded hash mismatch: {entry["path"]}; no file installed')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return 'imported' if source_root is not None else 'downloaded'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['test', 'references', 'all'], default='test')
    parser.add_argument('--from-directory', type=Path, help='Import a matching existing project data tree')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    failures = []
    for entry in manifest['files']:
        if args.group != 'all' and entry['group'] != args.group:
            continue
        try:
            status = install(entry, ROOT, args.from_directory, args.verify_only)
            print(f'{status}: {entry["path"]}')
        except (HTTPError, URLError, OSError, ValueError) as error:
            failures.append(entry['path'])
            print(f'UNAVAILABLE: {entry["path"]}: {error}', file=sys.stderr)
    if failures:
        print(f'{len(failures)} file(s) unavailable or mismatched. See data/README.md for historical-source access and verified local import.', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
