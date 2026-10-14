#!/usr/bin/env python3
"""Check tracked release files, local documentation links, and required assets."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    errors = []
    required = ['README.md', 'CITATION.cff', 'results.json', 'data/manifest.json',
                'papers/task1/main.pdf', 'papers/task2/main.pdf', 'papers/task3/main.pdf',
                'papers/conference/FinNLP65ADMLPoster.pdf', 'papers/conference/FinNLP65ADMLVideo.mp4']
    for relative in required:
        if relative not in files:
            errors.append(f'Required release asset not tracked: {relative}')
    for relative in filter(None, files):
        path = ROOT/relative
        if (relative.startswith(('data/raw/', 'data/external/', '.venv/', 'work/'))
                or path.suffix in {'.parquet', '.safetensors', '.pt', '.pth', '.pem', '.key', '.p12', '.pfx', '.keystore', '.credentials'}
                or path.name in {'.env', '.netrc', '.npmrc', '.pypirc', 'credentials.json', 'hosts.yml'}
                or path.name.startswith(('.env.', 'id_rsa', 'id_ed25519'))
                or path.is_symlink()):
            errors.append(f'Unexpected local/data asset in release: {relative}')
        if path.stat().st_size >= 50_000_000:
            errors.append(f'Oversized release file: {relative}')
        if path.suffix == '.md':
            for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if link.startswith(('https://', 'http://', 'mailto:', '#')):
                    continue
                target = unquote(link.split('#')[0])
                if not (path.parent/target).exists():
                    errors.append(f'Broken local link in {relative}: {target}')
        if path.suffix in {'.md', '.tex', '.json', '.yml', '.yaml', '.cff'}:
            if re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', path.read_text()):
                errors.append(f'Contact email in public document: {relative}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Release structure and local Markdown links passed ({len(list(filter(None, files)))} tracked files).')


if __name__ == '__main__':
    main()
