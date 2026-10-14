import csv
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_selected_runner_works_outside_repository_and_preserves_ids(tmp_path):
    responses = [
        '取締役会で承認しました。',
        '条件に合う企業があれば前向きに検討します。',
        '複数の前提を置いて検証しています。',
        '今年中の参入は見送る方向です。',
        '当社はこの市場へ参入しません。',
    ]
    ids = [81, 12, 104, 7, 63]
    pd.DataFrame({'id': ids, 'query': [
        f'Financial Question: 方針は？ Company Response: {response} Directly output'
        for response in responses
    ]}).to_parquet(tmp_path/'input.parquet')
    subprocess.run([
        sys.executable, str(ROOT/'scripts/run_final_submission.py'),
        '--task', '2', '--input', 'input.parquet', '--output', 'result.csv',
        '--work-dir', 'scratch', '--device', 'cpu',
    ], cwd=tmp_path, check=True, capture_output=True, text=True)
    with (tmp_path/'result.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row['id'] for row in rows] == list(map(str, ids))
    assert [row['prediction'] for row in rows] == ['+2', '+1', '0', '-1', '-2']
