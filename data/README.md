# Data sources and integrity

This repository contains a [download manifest](manifest.json), not benchmark corpora. Each entry records its expected path, source URL, purpose, and SHA-256. Organizer test inputs are pinned to the recorded Hugging Face revisions. The historical reference files are pinned by file hash; the July PolyFiQA repositories also have recorded revisions.

```bash
python scripts/download_data.py --group test
python scripts/download_data.py --group references
python scripts/download_data.py --group all --verify-only
```

As checked on **15 October 2026**, the three organizer test repositories are publicly accessible. Anonymous requests to the historical TheFinAI Plutus and July PolyFiQA endpoints returned HTTP 401. This release does not assume that those endpoints are still publicly downloadable. The downloader fails with an actionable message if a source is unavailable or a hash differs.

An existing dataset tree can be imported without a network request:

```bash
python scripts/download_data.py --group all --from-directory /path/to/existing/project
```

The source directory must contain the same `data/raw/...` and `data/external/...` relative paths; every file is checked before it is installed. Obtain restricted files from the relevant maintainer under the applicable terms. The Japanese selected rubric only needs its public organizer test packet and is independently runnable.

## Attribution

The historical references originate from **TheFinAI**: Plutus numeric/textual Greek NER and PolyFiQA Easy/Expert July snapshots. The official test packets are provided by **FinNLP-Multilingual-Understanding**. Dataset papers are cited in [the system-paper bibliography](../papers/common/references.bib).

The July PolyFiQA cards did not declare a license when originally audited; the base card and overview paper also gave inconsistent license descriptions. This project does not resolve that inconsistency by assigning its own license. Data and model rights remain with the respective providers, and reference files are not redistributed here.
