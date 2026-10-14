#!/usr/bin/env python3
"""Encode JF-ICR question/answer pairs with a Japanese-native encoder."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


MODEL = "cl-nagoya/ruri-v3-310m"


def main() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    parsed = frame["query"].map(parse_icr_query)
    # Ruri's model card specifies the topic prefix for classification and
    # clustering. Keep question and company response visibly separated.
    texts = [
        f"トピック: 質問: {item.question}\n会社回答: {item.response}"
        for item in parsed
    ]
    model = SentenceTransformer(MODEL, device="cuda")
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    output = ROOT / "data/derived/icr_ruri_v3_310m_embeddings.npy"
    np.save(output, embeddings)
    print(f"wrote {embeddings.shape} {embeddings.dtype} to {output}")


if __name__ == "__main__":
    main()
