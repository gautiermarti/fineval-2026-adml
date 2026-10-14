#!/usr/bin/env python3
"""Recover URL-addressed ATHEX reports from the collector provenance ledger."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import urllib.parse

import requests

from collect_athex_corpus import is_greek_financial_report


ROOT = Path("data/raw/finance_corpus")


def target_path(output: Path, issuer: str, url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    return output / f"{issuer}_{digest}.txt"


def recover(record: dict, output: Path, timeout: int) -> dict:
    url = record["url"]
    destination = target_path(output, str(record["issuer"]), url)
    if destination.exists() and is_greek_financial_report(
        destination.read_text(errors="ignore")
    ):
        return {**record, "status": "existing", "output": str(destination)}
    proxy = "https://r.jina.ai/http://" + url.split("://", 1)[-1]
    try:
        response = requests.get(
            proxy, timeout=timeout,
            headers={"User-Agent": "fineval-research-corpus/1.0"},
        )
        text = response.text
    except requests.RequestException as error:
        return {**record, "status": "request_error", "error": str(error)}
    if not is_greek_financial_report(text):
        return {
            **record, "status": "quality_reject", "bytes": len(text),
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return {
        **record, "status": "recovered", "output": str(destination),
        "bytes": len(text), "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--ledger", type=Path, default=ROOT / "athex_collection_progress.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "text/athex_url_recovery",
    )
    parser.add_argument("--domain", action="append")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "athex_recovery_manifest.jsonl",
    )
    args = parser.parse_args()
    by_url = {}
    for line in args.ledger.read_text().splitlines():
        record = json.loads(line)
        by_url.setdefault(record["url"], {"issuer": str(record["issuer"]), "url": record["url"]})
    records = list(by_url.values())
    if args.domain:
        domains = set(args.domain)
        records = [
            record for record in records
            if urllib.parse.urlparse(record["url"]).netloc in domains
        ]
    records = records[args.offset:]
    if args.limit is not None:
        records = records[:args.limit]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(recover, record, args.output, args.timeout) for record in records]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % 25 == 0 or completed == len(futures):
                counts = {}
                for row in results:
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                print(f"completed={completed}/{len(futures)} status={counts}", flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as handle:
        for result in sorted(results, key=lambda row: row["url"]):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
