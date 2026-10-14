#!/usr/bin/env python3
"""Record the effective FinEval organizer schedule and submission link.

The organizer page is a self-unpacking bundle.  Its embedded application
template still contains older dates, while small ``replaceTemplateSection``
calls in the outer bundle update what a browser actually renders.  This audit
applies the simple string overrides before extracting operational information.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


DEFAULT_URL = "https://www.finnlp2026sharedtask.info/"

TEMPLATE_PATTERN = re.compile(
    r'<script\s+type="__bundler/template">\s*(.*?)\s*</script>', re.DOTALL
)
STRING_OVERRIDE_PATTERN = re.compile(
    r"replaceTemplateSection\(\s*"
    r"(?P<q1>['\"`])(?P<before>.*?)(?P=q1)\s*,\s*"
    r"(?P<q2>['\"`])(?P<after>.*?)(?P=q2)\s*,\s*"
    r"(?P<q3>['\"`])(?P<label>.*?)(?P=q3)\s*\);",
    re.DOTALL,
)
DATE_PATTERN = re.compile(
    r'\{\s*date:\s*"([^"]+)",\s*event:\s*"([^"]+)",'
    r'\s*status:\s*"([^"]+)"\s*\}'
)
DRIVE_PATTERN = re.compile(
    r'link:\s*"(https://drive\.google\.com/drive/folders/[^"?]+)'
    r'(?:\?[^\"]*)?"'
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def effective_template(bundle_html: str) -> tuple[str, list[dict[str, Any]]]:
    """Decode the embedded template and apply browser-visible string patches."""
    match = TEMPLATE_PATTERN.search(bundle_html)
    if match is None:
        raise ValueError("organizer bundle has no __bundler/template payload")
    try:
        template = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ValueError("organizer template is not valid JSON") from error
    if not isinstance(template, str):
        raise ValueError("organizer template JSON is not a string")

    overrides: list[dict[str, Any]] = []
    for override in STRING_OVERRIDE_PATTERN.finditer(bundle_html[:match.start()]):
        before = override.group("before")
        after = override.group("after")
        occurrences = template.count(before)
        if occurrences:
            template = template.replace(before, after, 1)
        overrides.append({
            "label": override.group("label"),
            "before_found": occurrences > 0,
            "before_occurrences": occurrences,
        })
    return template, overrides


def parse_effective_site(bundle_html: str) -> dict[str, Any]:
    template, overrides = effective_template(bundle_html)
    dates = [
        {"date": date, "event": event, "status": status}
        for date, event, status in DATE_PATTERN.findall(template)
    ]
    if not dates:
        raise ValueError("no effective schedule entries found")

    submission_start = template.find('id="submission"')
    submission_end = template.find('id="leaderboard"', submission_start + 1)
    if submission_start < 0 or submission_end < 0:
        raise ValueError("submission section is missing")
    submission = template[submission_start:submission_end]
    solution_start = submission.find("Solution submission")
    paper_start = submission.find("System paper submission")
    if paper_start < 0:
        raise ValueError("system paper submission subsection is missing")
    solution = (
        submission[solution_start:paper_start] if solution_start >= 0 else ""
    )
    links = re.findall(r'href="(https?://[^"]+)"', solution)
    direct_submission_url = links[0] if links else None
    task_submission_urls = list(dict.fromkeys(re.findall(
        r'href="(https://huggingface\.co/spaces/[^"]+)"', submission
    )))
    submission_urls = list(dict.fromkeys(
        ([direct_submission_url] if direct_submission_url else [])
        + task_submission_urls
    ))
    submission_url = submission_urls[0] if submission_urls else None
    submission_status = "available" if submission_urls else (
        "TBD" if "TBD" in solution else "unresolved"
    )

    drive_urls = DRIVE_PATTERN.findall(template)
    if len(drive_urls) != 3:
        raise ValueError(f"expected three organizer Drive folders, found {len(drive_urls)}")

    event_dates = {row["event"]: row["date"] for row in dates}
    return {
        "dates": dates,
        "event_dates": event_dates,
        "submission_status": submission_status,
        "submission_url": submission_url,
        "submission_urls": submission_urls,
        "task_submission_urls": task_submission_urls,
        "drive_folder_urls": drive_urls,
        "applied_string_overrides": overrides,
        "all_parsed_overrides_found": all(
            row["before_found"] for row in overrides
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-test-release")
    parser.add_argument("--require-submission-link", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    headers: dict[str, str] = {}
    if args.html is not None:
        raw = args.html.read_bytes()
        source = str(args.html)
    else:
        # curl is used deliberately: on the DGX host its TLS/network path is
        # the same one used by the release runbook, while Python's urllib3
        # path can stall before honoring its read timeout on this Netlify site.
        response = subprocess.run(
            [
                "curl", "--location", "--silent", "--show-error",
                "--fail-with-body", "--max-time", str(args.timeout), args.url,
            ],
            check=True,
            capture_output=True,
        )
        raw = response.stdout
        source = args.url

    parsed = parse_effective_site(raw.decode("utf-8"))
    failures = []
    actual_release = parsed["event_dates"].get(
        "Test set release, test phase starts"
    )
    if args.expect_test_release and actual_release != args.expect_test_release:
        failures.append(
            f"test release is {actual_release!r}, expected {args.expect_test_release!r}"
        )
    if args.require_submission_link and not parsed["submission_urls"]:
        failures.append("solution submission link is not available")
    if not parsed["all_parsed_overrides_found"]:
        failures.append("at least one parsed site override did not match the template")

    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_sha256": sha256_bytes(raw),
        "response_headers": headers,
        **parsed,
        "failures": failures,
        "status": "pass" if not failures else "attention_required",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "test_release": actual_release,
        "submission_status": parsed["submission_status"],
        "submission_url": parsed["submission_url"],
        "submission_urls": parsed["submission_urls"],
        "output": str(args.output),
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
