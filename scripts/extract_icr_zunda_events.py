#!/usr/bin/env python3
"""Extract label-free Zunda event/modality views for JF-ICR rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.icr_events import event_candidates  # noqa: E402
from fineval.icr_zunda import parse_zunda_block, split_zunda_blocks  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(command: list[str], text: str, env: dict[str, str], ok=(0,)) -> str:
    result = subprocess.run(
        command, input=text, text=True, capture_output=True, env=env, check=False,
    )
    if result.returncode not in ok:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet",
    )
    parser.add_argument(
        "--prefix", type=Path,
        default=ROOT / "artifacts/tools/japanese_modality/prefix",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/icr_zunda_events_v1",
    )
    args = parser.parse_args()

    cabocha = args.prefix / "bin/cabocha"
    zunda = args.prefix / "bin/zunda"
    cabocharc = args.prefix / "etc/cabocharc"
    mecabrc = args.prefix / "usr/etc/mecabrc"
    model = args.prefix / "share/zunda/model_ipa"
    dictionary = args.prefix / "share/zunda/dic"
    functional_model = model / "funcsem_v2.2.2.crfpp.model"
    required = (cabocha, zunda, cabocharc, mecabrc, functional_model)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing local Zunda toolchain: {missing}")
    if functional_model.stat().st_size < 700_000_000:
        raise RuntimeError("Zunda functional model is still a Git-LFS pointer")

    frame = pd.read_parquet(args.input)
    records = []
    flat_candidates = []
    for row in frame.itertuples(index=False):
        parsed = parse_icr_query(row.query)
        candidates = [candidate.to_dict() for candidate in event_candidates(
            parsed.question, parsed.response,
        )]
        records.append({
            "id": int(row.id), "question": parsed.question,
            "response": parsed.response, "candidates": candidates,
        })
        flat_candidates.extend(candidates)

    # CaboCha and Zunda both accept one sentence per line and delimit outputs by
    # EOS.  Analyze all candidates in one process so the 759 MB model loads once.
    analysis_texts = [
        " ".join(str(candidate["text"]).splitlines()) for candidate in flat_candidates
    ]
    batch = "\n".join(analysis_texts) + "\n"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, (
        str(args.prefix / "lib"),
        str(args.prefix / "usr/lib/aarch64-linux-gnu"),
        env.get("LD_LIBRARY_PATH", ""),
    )))
    env["MECABRC"] = str(mecabrc)
    cabocha_output = run_checked(
        [str(cabocha), "-r", str(cabocharc), "-f1"], batch, env,
    )
    zunda_output = run_checked(
        [str(zunda), "-i", "1", "-m", str(model), "-d", str(dictionary)],
        cabocha_output, env, ok=(0, 1),
    )
    blocks = split_zunda_blocks(zunda_output)
    if len(blocks) != len(flat_candidates):
        raise RuntimeError(
            f"expected {len(flat_candidates)} Zunda blocks, received {len(blocks)}"
        )
    for candidate, block in zip(flat_candidates, blocks):
        events, functional = parse_zunda_block(block)
        candidate["zunda_events"] = [event.to_dict() for event in events]
        candidate["functional_expressions"] = functional

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "events.jsonl"
    output.write_text("".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    ))
    event_count = sum(
        len(candidate["zunda_events"])
        for record in records for candidate in record["candidates"]
    )
    manifest = {
        "rows": len(records),
        "candidates": len(flat_candidates),
        "zunda_events": event_count,
        "candidate_texts_are_input_only": True,
        "jf_icr_labels_read": False,
        "current_or_hidden_test_used": False,
        "toolchain": {
            "zunda_version": run_checked([str(zunda), "--version"], "", env, ok=(0, 1)).strip(),
            "cabocha_version": run_checked([str(cabocha), "--version"], "", env).strip(),
            "functional_model_bytes": functional_model.stat().st_size,
            "functional_model_sha256": sha256(functional_model),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
