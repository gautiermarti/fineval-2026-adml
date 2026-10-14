import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_icr_edinet_mlm import company_split, excluded_company, token_blocks
from adapt_icr_edinet_mlm import BlockDataset
from annotate_icr_edinet_outlook import cue_count


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}

    def build_inputs_with_special_tokens(self, tokens):
        return [101, *tokens, 102]


def test_benchmark_company_exclusion_is_conservative():
    assert excluded_company("三菱商事株式会社")
    assert excluded_company("株式会社野村総合研究所")
    assert excluded_company("住友商事株式会社")
    assert excluded_company("株式会社三井住友フィナンシャルグループ")
    assert not excluded_company("株式会社エディオン")


def test_company_split_is_stable():
    assert company_split("株式会社エディオン") == company_split("株式会社エディオン")
    assert company_split("株式会社エディオン") in {"train", "validation"}


def test_token_blocks_drop_short_tail_and_add_special_tokens():
    blocks = token_blocks(TinyTokenizer(), "x" * 11, 8)
    assert blocks == [[101, 0, 1, 2, 3, 4, 5, 6, 7, 102], [101, 8, 9, 10, 102]]


def test_block_dataset_normalizes_arrow_numpy_integers():
    import numpy as np

    item = BlockDataset([np.asarray([2, 3, 4], dtype=np.int64)])[0]
    assert item == {"input_ids": [2, 3, 4]}
    assert all(type(token) is int for token in item["input_ids"])


def test_outlook_cue_count_counts_distinct_patterns():
    assert cue_count([1, 2, 1, 2, 4], [[1], [1, 2], [3], [2, 4]]) == 3
