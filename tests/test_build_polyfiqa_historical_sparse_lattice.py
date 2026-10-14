from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_polyfiqa_historical_sparse_lattice import partition_name  # noqa: E402


def test_partition_name_uses_exact_discovery_fingerprint() -> None:
    discovery = {"known"}
    assert partition_name("known", discovery) == "late_discovery"
    assert partition_name("other", discovery) == "early_confirmation"
