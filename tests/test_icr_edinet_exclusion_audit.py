import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_icr_edinet_exclusions import SOURCE_IDENTITIES


def test_every_reconstructed_source_group_has_an_identity():
    source = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")
    assert set(source.source_group) == set(SOURCE_IDENTITIES)
