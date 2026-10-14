import hashlib
from pathlib import Path

import pytest

from scripts.download_data import destination, install


def test_import_is_hash_checked_and_preserves_existing_file(tmp_path):
    source = tmp_path/'source'
    target = tmp_path/'target'
    relative = 'data/raw/example.parquet'
    fixture = source/relative
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b'fixture')
    entry = {'path': relative, 'sha256': hashlib.sha256(b'fixture').hexdigest()}
    assert install(entry, target, source) == 'imported'
    assert install(entry, target, verify_only=True) == 'verified'
    (target/relative).write_bytes(b'different source version')
    with pytest.raises(ValueError, match='different hash'):
        install(entry, target, source)
    assert (target/relative).read_bytes() == b'different source version'


def test_wrong_source_is_never_installed(tmp_path):
    source, target = tmp_path/'source', tmp_path/'target'
    fixture = source/'data/raw/example.parquet'
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b'wrong')
    entry = {'path': 'data/raw/example.parquet', 'sha256': hashlib.sha256(b'right').hexdigest()}
    with pytest.raises(ValueError, match='hash mismatch'):
        install(entry, target, source)
    assert not (target/entry['path']).exists()


def test_manifest_paths_cannot_escape_dataset_tree(tmp_path):
    for value in ['../outside', 'data/../../outside', '/etc/passwd', 'README.md']:
        with pytest.raises(ValueError):
            destination(tmp_path, value)
