#!/usr/bin/env python3
"""Collect public Japanese investor-relations Q&A from unrelated issuers.

The collector is deliberately allow-list based.  It records source URLs and
content hashes, extracts only explicit question/answer turns, and removes
exact or near duplicates of the released JF-ICR benchmark before writing the
silver-label candidate corpus.
"""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urljoin, urlparse
import json
import re
import subprocess
import sys
import time

from bs4 import BeautifulSoup
import pandas as pd
from rapidfuzz.fuzz import ratio
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.parsing import parse_icr_query  # noqa: E402


USER_AGENT = "ADML-FinNLP-research/1.0 (public Japanese IR Q&A corpus)"
QA_CUES = (
    "質疑", "質問", "q&a", "q_and_a", "questions", "question", "_qa", "/qa", "qa_",
)
BENCHMARK_ISSUER_TERMS = (
    "ana", "chiyoda", "dnp", "itochu", "kajima", "mufg", "mitsubishi",
    "mitsui", "nissan", "nomura", "obayashi", "shimizu", "softbank",
    "sumitomo", "三菱", "三井", "伊藤忠", "住友", "野村", "日産", "鹿島",
    "大林", "清水建設", "大日本印刷", "全日本空輸",
)


@dataclass(frozen=True)
class Source:
    issuer: str
    index_urls: tuple[str, ...]
    direct_urls: tuple[str, ...] = ()


SOURCES = (
    Source(
        "fast_retailing",
        (
            "https://www.fastretailing.com/jp/ir/library/earning.html",
            "https://www.fastretailing.com/jp/ir/library/earning_back.html",
        ),
    ),
    Source(
        "nikon",
        ("https://www.jp.nikon.com/company/ir/ir_library/result/",),
    ),
    Source(
        "chubu_electric",
        (
            "https://www.chuden.co.jp/ir/ir_siryo/kessan/",
            "https://www.chuden.co.jp/ir/ir_siryo/kessan/backnumber/",
        ),
    ),
    Source(
        "shizuoka_financial_group",
        (
            "https://www.shizuoka-fg.co.jp/ir/ir-presentation.html",
            "https://www.shizuoka-fg.co.jp/ir/2025/ir-presentation.html",
            "https://www.shizuoka-fg.co.jp/ir/2024/ir-presentation.html",
            "https://www.shizuoka-fg.co.jp/ir/2023/ir-presentation.html",
        ),
    ),
    Source(
        "estic",
        (
            "https://www.estic.co.jp/ir/presentations/",
            "https://www.estic.co.jp/ir/presentations/archive/index.html",
        ),
    ),
    Source(
        "tokyo_electron_device",
        ("https://www.teldevice.co.jp/ir/",),
        tuple(
            f"https://www.teldevice.co.jp/ir/qa{year}{suffix}.html"
            for year in range(2018, 2027)
            for suffix in ("", "b")
        ),
    ),
    Source(
        "oji_holdings",
        ("https://www.ojiholdings.co.jp/ir/library/result/",),
    ),
    Source(
        "tohoku_electric",
        ("https://www.tohoku-epco.co.jp/ir_n/report/co_explanation/archive.html",),
    ),
    Source(
        "tokyo_ink",
        ("https://www.tokyoink.co.jp/ir/news/detail/20251127112158.html",),
    ),
    Source(
        "infroneer_holdings",
        ("https://www.infroneer.com/jp/news-article/2026/06/04/1223.html",),
    ),
    Source(
        "daidan",
        (),
        ("https://www.daidan.co.jp/wp/wp-content/uploads/2025/05/1703e3ad830de3cef7b9249eaf80d8f1.pdf",),
    ),
    Source(
        "credit_saison",
        (),
        ("https://corporate.saisoncard.co.jp/ir/pdf/ManagementReport_FY2022.pdf",),
    ),
    Source(
        "iij",
        ("https://www.iij.ad.jp/ir/library/questions/",),
    ),
    Source(
        "iyogin_holdings",
        tuple(
            f"https://www.iyogin-hd.co.jp/ir/presentation/presentation{year}.html"
            for year in range(2018, 2027)
        ),
    ),
    Source(
        "ibiden",
        ("https://www.ibiden.co.jp/ir/library/presentation/",),
    ),
    Source(
        "septeni_holdings",
        ("https://www.septeni-holdings.co.jp/ir/library/briefing/",),
    ),
    Source(
        "sojitz",
        ("https://www.sojitz.com/jp/ir/meetings/financial/",),
    ),
    Source(
        "minebea_mitsumi",
        tuple(
            f"https://www.minebeamitsumi.com/corp/investors/disclosure/financial/p{year}/"
            for year in range(2018, 2027)
        ),
    ),
    Source(
        "sobal",
        ("https://www.sobal.co.jp/ir/irlibrary/kessan/",),
    ),
    Source(
        "nitto_denko",
        ("https://www.nitto.com/jp/ja/ir/library/qa/",),
    ),
    Source(
        "nhk_spring",
        (),
        ("https://www.nhkspg.co.jp/hubfs/nhkspg.co.jp/pdf/ir/library/meeting/2026/2026_03_QA.pdf",),
    ),
    Source(
        "denka",
        (),
        ("https://www.denka.co.jp/storage/news/pdf/1251/20240809_denka_ir_conference_qa.pdf",),
    ),
    Source(
        "daishinku_6962",
        (),
        ("https://pdf.irpocket.com/C6962/PDLX/MhDQ/mkie.pdf",),
    ),
    Source(
        "shinsei_switch_1952",
        (),
        ("https://pdf.irpocket.com/C1952/hZTq/TClm/Kggc.pdf",),
    ),
    Source(
        "organo_6368",
        (),
        ("https://pdf.irpocket.com/C6368/hZTq/gqo4/Q0ns.pdf",),
    ),
    Source(
        "hokuetsu_8377",
        (),
        ("https://pdf.irpocket.com/C8377/PDLX/xFBj/Pbcj.pdf",),
    ),
    Source(
        "daiwa_ir",
        (),
        ("https://www.daiwair.co.jp/e-cast_files/7035/pdf/meeting_memo_pdf.pdf",),
    ),
)


@dataclass(frozen=True)
class Document:
    issuer: str
    url: str
    final_url: str
    content_type: str
    sha256: str
    bytes: int
    retrieval_utc: str
    extraction_method: str
    text_chars: int
    status: str
    error: str = ""


def normalize(value: str) -> str:
    value = value.replace("\u3000", " ").replace("\ufeff", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def benchmark_pairs() -> list[tuple[str, str]]:
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    result = []
    for query in frame["query"]:
        example = parse_icr_query(query)
        result.append((normalize(example.question), normalize(example.response)))
    return result


def is_benchmark_issuer(issuer: str, url: str) -> bool:
    folded = f"{issuer} {urlparse(url).netloc} {url}".casefold()
    return any(term.casefold() in folded for term in BENCHMARK_ISSUER_TERMS)


def request(session: requests.Session, url: str, attempts: int = 3) -> requests.Response:
    last_error = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=(15, 60), allow_redirects=True)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def html_text(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for node in soup.select("script, style, noscript, svg, nav, header, footer"):
        node.decompose()
    main = soup.select_one("main, #contents, #content, .contents, .content") or soup.body or soup
    return normalize(main.get_text("\n", strip=True))


def pdf_text(content: bytes) -> tuple[str, str]:
    with TemporaryDirectory(prefix="adml_irqa_") as directory:
        pdf = Path(directory) / "source.pdf"
        text = Path(directory) / "source.txt"
        pdf.write_bytes(content)
        completed = subprocess.run(
            ["pdftotext", "-layout", "-nopgbrk", str(pdf), str(text)],
            capture_output=True, text=True, check=False, timeout=120,
        )
        if completed.returncode != 0:
            raise ValueError(f"pdftotext failed: {completed.stderr.strip()}")
        extracted = normalize(text.read_text(errors="replace"))
    return extracted, "pdftotext-layout"


def candidate_links(base_url: str, content: bytes) -> set[str]:
    soup = BeautifulSoup(content, "html.parser")
    result = set()
    base_domain = urlparse(base_url).netloc.casefold()
    for anchor in soup.select("a[href]"):
        href = urljoin(base_url, anchor.get("href", "")).split("#", 1)[0]
        if not href.startswith(("http://", "https://")):
            continue
        text = normalize(anchor.get_text(" ", strip=True))
        folded = f"{text} {href}".casefold()
        # Shared IR hosting is allowed only when explicitly linked from the
        # issuer's own page and the anchor itself identifies Q&A material.
        same_domain = urlparse(href).netloc.casefold() == base_domain
        explicit_anchor = any(cue in text.casefold() for cue in QA_CUES)
        if "faq" in urlparse(href).path.casefold() and not any(
            cue in text for cue in ("決算", "説明会", "質疑")
        ):
            continue
        if any(cue in folded for cue in QA_CUES) and (same_domain or explicit_anchor):
            result.add(href)
    return result


MARKER_PATTERNS = (
    re.compile(
        r"(?:^|\n)\s*(?:【|〖|＜|<)?\s*(?:質問|Question)\s*(?:[】〗＞>]|[0-9０-９]*\s*[：:．.]?)\s*"
        r"(?P<question>.*?)\s*(?:\n|^)\s*(?:【|〖|＜|<)?\s*(?:回答|Answer)\s*"
        r"(?:[】〗＞>]|[0-9０-９]*\s*[：:．.]?)\s*(?P<answer>.*?)"
        r"(?=(?:\n\s*(?:【|〖|＜|<)?\s*(?:質問|Question)\b)|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*[QＱ](?:(?:[ \t]*[0-9０-９]+[ \t]*[：︓:．.\-]?)|"
        r"(?:[ \t]*[：︓:．.\-])|(?:[ \t]+))\s*(?P<question>.*?)\s*"
        r"(?:\n|^)\s*[AＡ](?:(?:[ \t]*[0-9０-９]+[ \t]*[：︓:．.\-]?)|"
        r"(?:[ \t]*[：︓:．.\-])|(?:[ \t]+))\s*(?P<answer>.*?)"
        r"(?=(?:\n\s*[QＱ](?:(?:[ \t]*[0-9０-９]+[ \t]*[：︓:．.\-]?)|"
        r"(?:[ \t]*[：︓:．.\-])|(?:[ \t]+)))|\Z)",
        re.DOTALL,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:問|質問)\s*[0-9０-９]+\s*[：:．.\-]?\s*(?P<question>.*?)\s*"
        r"(?:\n|^)\s*(?:答|回答)\s*[0-9０-９]*\s*[：:．.\-]?\s*(?P<answer>.*?)"
        r"(?=(?:\n\s*(?:問|質問)\s*[0-9０-９]+)|\Z)",
        re.DOTALL,
    ),
)


def clean_segment(value: str) -> str:
    value = re.sub(r"\n\s*(?:Copyright|©).*", "", value, flags=re.I | re.S)
    value = re.sub(r"\n\s*[-―─]{3,}.*", "", value, flags=re.S)
    return normalize(value).strip(" -―─:：")


def japanese_fraction(value: str) -> float:
    if not value:
        return 0.0
    japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", value))
    return japanese / max(len(value), 1)


def extract_pairs(text: str) -> list[tuple[str, str]]:
    candidates = []
    for pattern in MARKER_PATTERNS:
        for match in pattern.finditer(text):
            question = clean_segment(match.group("question"))
            answer = clean_segment(match.group("answer"))
            if not (8 <= len(question) <= 1_500 and 15 <= len(answer) <= 8_000):
                continue
            if japanese_fraction(question + answer) < 0.08:
                continue
            candidates.append((question, answer))
    # A document can match more than one permissive marker grammar.
    unique = {}
    for question, answer in candidates:
        key = sha256((normalize(question) + "\x1f" + normalize(answer)).encode()).hexdigest()
        unique.setdefault(key, (question, answer))
    return list(unique.values())


def benchmark_similarity(
    question: str, answer: str, released: list[tuple[str, str]],
) -> tuple[float, int]:
    best, best_index = 0.0, -1
    for index, (gold_question, gold_answer) in enumerate(released):
        score = max(ratio(question, gold_question), ratio(answer, gold_answer)) / 100
        if score > best:
            best, best_index = score, index
    return best, best_index


def save_raw(root: Path, issuer: str, digest: str, suffix: str, content: bytes) -> Path:
    path = root / "raw" / issuer / f"{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    return path


def collect_document(
    session: requests.Session, issuer: str, url: str, output: Path,
) -> tuple[Document, str]:
    retrieved = datetime.now(timezone.utc).isoformat()
    try:
        response = request(session, url)
        content = response.content
        digest = sha256(content).hexdigest()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].casefold()
        is_pdf = content.startswith(b"%PDF") or "pdf" in content_type or response.url.casefold().endswith(".pdf")
        if is_pdf:
            save_raw(output, issuer, digest, ".pdf", content)
            text, method = pdf_text(content)
        else:
            save_raw(output, issuer, digest, ".html", content)
            text, method = html_text(content), "beautifulsoup-main-text"
        text_path = output / "text" / issuer / f"{digest}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        if not text_path.exists():
            text_path.write_text(text)
        status = "ok" if len(text) >= 50 else "empty_text"
        return Document(
            issuer, url, response.url, content_type, digest, len(content), retrieved,
            method, len(text), status,
        ), text
    except (requests.RequestException, ValueError, subprocess.SubprocessError, OSError) as exc:
        return Document(
            issuer, url, url, "", "", 0, retrieved, "", 0, "error",
            f"{type(exc).__name__}: {exc}",
        ), ""


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/external/japanese_ir_qa",
    )
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    released = benchmark_pairs()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.5"})
    documents: list[Document] = []
    targets: list[tuple[str, str]] = []

    for source in SOURCES:
        for index_url in source.index_urls:
            if is_benchmark_issuer(source.issuer, index_url):
                continue
            document, text = collect_document(session, source.issuer, index_url, args.output)
            documents.append(document)
            if document.status == "ok":
                raw = (args.output / "raw" / source.issuer / f"{document.sha256}.html").read_bytes()
                targets.extend((source.issuer, link) for link in candidate_links(document.final_url, raw))
                # Some issuers publish the Q&A directly on their index page.
                if extract_pairs(text):
                    targets.append((source.issuer, index_url))
            time.sleep(args.delay)
        targets.extend((source.issuer, url) for url in source.direct_urls)

    # Preserve discovery order while removing repeated links and index fetches.
    unique_targets = []
    observed = set()
    already_fetched = {(item.issuer, item.url) for item in documents if item.status == "ok"}
    for issuer, url in targets:
        key = (issuer, url)
        if key in observed or is_benchmark_issuer(issuer, url):
            continue
        observed.add(key)
        if key in already_fetched:
            continue
        unique_targets.append(key)
    if args.max_documents is not None:
        unique_targets = unique_targets[:args.max_documents]

    texts: dict[tuple[str, str], tuple[Document, str]] = {}
    for issuer, url in unique_targets:
        document, text = collect_document(session, issuer, url, args.output)
        documents.append(document)
        texts[(issuer, url)] = (document, text)
        time.sleep(args.delay)

    # Include successfully fetched index pages that themselves contain pairs.
    for document in documents:
        if document.status != "ok" or (document.issuer, document.url) in texts:
            continue
        text_path = args.output / "text" / document.issuer / f"{document.sha256}.txt"
        texts[(document.issuer, document.url)] = (document, text_path.read_text())

    pairs = []
    seen_pair_hashes = set()
    excluded_near_duplicates = []
    for (issuer, url), (document, text) in texts.items():
        for pair_index, (question, answer) in enumerate(extract_pairs(text)):
            pair_hash = sha256((normalize(question) + "\x1f" + normalize(answer)).encode()).hexdigest()
            if pair_hash in seen_pair_hashes:
                continue
            similarity, benchmark_index = benchmark_similarity(question, answer, released)
            record = {
                "pair_id": pair_hash,
                "issuer": issuer,
                "source_url": document.final_url,
                "source_sha256": document.sha256,
                "source_pair_index": pair_index,
                "question": question,
                "response": answer,
                "benchmark_max_similarity": round(similarity, 6),
                "benchmark_nearest_id": benchmark_index,
                "license": "public-corporate-IR; reuse_terms_not_asserted",
            }
            if similarity >= 0.92:
                excluded_near_duplicates.append(record)
                continue
            seen_pair_hashes.add(pair_hash)
            pairs.append(record)

    pd.DataFrame([asdict(document) for document in documents]).to_csv(
        args.output / "manifest.csv", index=False,
    )
    for filename, records in (
        ("pairs.jsonl", pairs),
        ("excluded_benchmark_near_duplicates.jsonl", excluded_near_duplicates),
    ):
        with (args.output / filename).open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "documents_attempted": len(documents),
        "documents_ok": sum(item.status == "ok" for item in documents),
        "documents_error": sum(item.status == "error" for item in documents),
        "issuers_with_pairs": len({record["issuer"] for record in pairs}),
        "qa_pairs": len(pairs),
        "excluded_benchmark_near_duplicates": len(excluded_near_duplicates),
        "maximum_retained_benchmark_similarity": max(
            (record["benchmark_max_similarity"] for record in pairs), default=0.0,
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["documents_error"]:
        errors = [item for item in documents if item.status == "error"]
        print("errors:")
        for item in errors[:20]:
            print(item.issuer, item.url, item.error)


if __name__ == "__main__":
    main()
