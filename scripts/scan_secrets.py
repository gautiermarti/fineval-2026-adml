#!/usr/bin/env python3
"""Scan reachable Git history with a checksum-pinned Gitleaks binary."""
from pathlib import Path
import hashlib
import io
import json
import platform
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

VERSION = '8.30.1'
LINUX_ASSETS = {
    'aarch64': ('arm64', 'e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080'),
    'x86_64': ('x64', '551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb'),
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    if platform.system() != 'Linux' or platform.machine() not in LINUX_ASSETS:
        raise SystemExit('This pinned installer supports Linux x86_64/ARM64. On other platforms install Gitleaks and run: gitleaks git --redact --log-opts="--all"')
    arch, expected = LINUX_ASSETS[platform.machine()]
    url = f'https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}/gitleaks_{VERSION}_linux_{arch}.tar.gz'
    blob = urlopen(url, timeout=60).read()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise SystemExit('Scanner download failed checksum verification')
    with tempfile.TemporaryDirectory(prefix='fineval-secret-scan-') as temporary:
        folder = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as archive:
            member = next(m for m in archive.getmembers() if m.isfile() and Path(m.name).name == 'gitleaks')
            binary = folder/'gitleaks'
            binary.write_bytes(archive.extractfile(member).read())
            binary.chmod(0o755)
        config = folder/'config.toml'
        config.write_text('[extend]\nuseDefault = true\n')
        report = folder/'findings.json'
        result = subprocess.run([
            str(binary), 'git', str(ROOT), '--log-opts=--all', '--redact=100',
            '--no-banner', '--ignore-gitleaks-allow', '--max-archive-depth', '2',
            '--config', str(config), '--report-format', 'json', '--report-path', str(report),
        ], text=True, capture_output=True)
        findings = json.loads(report.read_text()) if report.exists() else []
        for finding in findings:
            # Never print matched values, even when a scan fails in public CI logs.
            print({key: finding.get(key) for key in ['RuleID', 'File', 'StartLine', 'Commit']})
        print(f'Gitleaks {VERSION}: {len(findings)} finding(s), exit status {result.returncode}')
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
