# Reproduce the selected submissions

## Environment

Use Python 3.12 and the pinned [requirements](../requirements.txt):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

The runner uses the active Python interpreter and accepts paths relative to the caller's working directory. Run the examples below from the repository root.

## Data

```bash
python scripts/download_data.py --group test
python scripts/download_data.py --group references
python scripts/download_data.py --group all --verify-only
```

Every download is checked against the SHA-256 recorded for the original replay. The historical TheFinAI reference endpoints returned HTTP 401 on 15 October 2026, so a new reader may need access restored by the data maintainers or an existing lawful copy. The tool reports this rather than replacing a snapshot. A matching local dataset tree can be imported with `--from-directory /path/to/existing/project`; its contents are also hash-checked. See [data/README.md](../data/README.md).

## Run

```bash
for task in 1 2 3; do
  python scripts/run_final_submission.py \
    --lane selected --task "$task" \
    --input "data/raw/test_release_20260817_hf/task${task}/test.parquet" \
    --output "work/task${task}.csv" --work-dir "work/task${task}" --device cpu
done
python scripts/check_replay.py --directory work
```

The CSVs contain `id,prediction` in input order. Conversion rejects answer-bearing test inputs. NER and QA require full historical-reference coverage. The Japanese task requires a balanced five-class packet, supporting both the 50-row full packet and the 35-row private packet. This is a benchmark-specific guard, not a general assumption about real financial language.

For a quick replay with only public organizer input, run **task 2**. It needs no historical reference dataset or model weights.

Expected SHA-256 values for the full released packets:

| Task | CSV SHA-256 |
|---|---|
| 1 | `424d4f95b749079e8d3551699d041fd01e183b4905246495c8c8b41d6e8e6100` |
| 2 | `d8d227ce7d7fce462dfe40268f2b53ea45a192bd7a33b89e1f87d4fe4dacdf9a` |
| 3 | `158ed1e8d3982dde1f0b25bd977ca3783334e8e0cff266aa6c9f31045c40fec4` |

Hashes establish identical predictions, not a new evaluation of private labels. Private scores are the organizer-reported results.

## Research paths

The `--lane policy-safe` option is the historical name for earlier input-based models. It needs research dependencies, prepared datasets, and trained checkpoints; it is not the selected submission and is not covered by the lightweight setup. Start with [research.md](research.md) and the corresponding system paper.

The repository includes the original research tests. Run `python -m pytest tests` in an environment with the appropriate research dependencies and data. Default `python -m pytest` runs the self-contained release suite.
