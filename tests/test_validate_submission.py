import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_task3_allows_repeated_packet_ids(tmp_path):
    source=tmp_path/'input.jsonl'; pred=tmp_path/'pred.jsonl'
    source.write_text('\n'.join(json.dumps({'task_id':'packet-1'}) for _ in range(2)))
    pred.write_text('\n'.join(json.dumps({'task_id':'packet-1','answer':'answer'}) for _ in range(2)))
    result=subprocess.run([sys.executable,str(ROOT/'scripts/validate_submission.py'),'--task','3','--input',str(source),'--predictions',str(pred)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert 'unique_task_id=false' in result.stdout

def test_task1_accepts_decimal_comma_and_empty_answer(tmp_path):
    source=tmp_path/'input.jsonl'; pred=tmp_path/'pred.jsonl'
    source.write_text('\n'.join(json.dumps({'text':str(i)}) for i in range(2)))
    pred.write_text('\n'.join([
        json.dumps({'row_index':0,'answer':'19,9%, ΠΟΣΟΣΤΑ'}),
        json.dumps({'row_index':1,'answer':''}),
    ]))
    result=subprocess.run([sys.executable,str(ROOT/'scripts/validate_submission.py'),'--task','1','--input',str(source),'--predictions',str(pred)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert 'unique_row_index=true' in result.stdout


def test_task1_rejects_cross_track_entity_type(tmp_path):
    source = tmp_path / 'input.jsonl'
    pred = tmp_path / 'pred.jsonl'
    source.write_text(json.dumps({
        'query': 'Αναγνώρισε αριθμητικές οντότητες και χρηματικά ποσά.',
        'text': 'Η αξία είναι 20 ευρώ.',
    }))
    pred.write_text(json.dumps({
        'row_index': 0, 'answer': 'εταιρεία, ΟΡΓΑΝΙΣΜΟΣ',
    }))
    result = subprocess.run([
        sys.executable, str(ROOT / 'scripts/validate_submission.py'),
        '--task', '1', '--input', str(source), '--predictions', str(pred),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'invalid for Task 1 numeric track' in result.stderr


def test_task1_rejects_redundant_inflected_calendar_surface(tmp_path):
    source = tmp_path / 'input.jsonl'
    pred = tmp_path / 'pred.jsonl'
    source.write_text(json.dumps({
        'query': 'Αναγνώρισε αριθμητικές οντότητες και χρηματικά ποσά.',
        'text': 'Η συνεδρίαση έγινε την 8ης Ιουνίου 2023.',
    }))
    pred.write_text(json.dumps({
        'row_index': 0,
        'answer': '8ης, ΧΡΟΝΙΚΑ\n8, ΧΡΟΝΙΚΑ\n2023, ΧΡΟΝΙΚΑ',
    }))
    result = subprocess.run([
        sys.executable, str(ROOT / 'scripts/validate_submission.py'),
        '--task', '1', '--input', str(source), '--predictions', str(pred),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'redundant inflected calendar-day surface' in result.stderr


def test_task1_accepts_numeric_calendar_core(tmp_path):
    source = tmp_path / 'input.jsonl'
    pred = tmp_path / 'pred.jsonl'
    source.write_text(json.dumps({
        'query': 'Αναγνώρισε αριθμητικές οντότητες και χρηματικά ποσά.',
        'text': 'Η συνεδρίαση έγινε την 8ης Ιουνίου 2023.',
    }))
    pred.write_text(json.dumps({
        'row_index': 0, 'answer': '8, ΧΡΟΝΙΚΑ\n2023, ΧΡΟΝΙΚΑ',
    }))
    result = subprocess.run([
        sys.executable, str(ROOT / 'scripts/validate_submission.py'),
        '--task', '1', '--input', str(source), '--predictions', str(pred),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
