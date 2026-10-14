#!/usr/bin/env python3
"""Verify prediction CSVs against the selected full-packet submission hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('work'))
    parser.add_argument('--task', type=int, choices=[1, 2, 3], action='append')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    results = json.loads((ROOT/'results.json').read_text())
    checks = []
    for task in args.task or [1, 2, 3]:
        path = args.directory/f'task{task}.csv'
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = results['tasks'][str(task)]['selected_csv_sha256']
        checks.append({'task': task, 'actual_sha256': actual, 'expected_sha256': expected, 'matches': actual == expected})
        print(f'Task {task}: {"PASS" if actual == expected else "FAIL"} {actual}')
    report = {'python': platform.python_version(), 'checks': checks, 'all_passed': all(c['matches'] for c in checks)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    if not report['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
