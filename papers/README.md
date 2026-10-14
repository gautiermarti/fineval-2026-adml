# ADML FinEval 2026 system papers

Authors on all three papers:

- Gautier Marti — ADIA
- Khaled AlNuaimi — ADIA; Khalifa University

Manuscripts:

1. `task1/main.pdf` — Greek Financial NER — private entity F1 0.9675926, rank 1.
2. `task2/main.pdf` — Japanese ICR — private accuracy 0.9714286 (34/35), tied best and displayed rank 1.
3. `task3/main.pdf` — Multilingual Financial QA — private ROUGE-1 F1 0.3726320, rank 2.

Each manuscript has four pages of main text. References and reproducibility appendices begin on page 5. The papers use the official ACL style and A4 paper size.

## Build

From the repository root:

```bash
bash papers/build.sh
```

The build uses Tectonic/XeTeX with TeX Gyre Termes. The sources also select the standard `times` package automatically under pdfLaTeX.

## Validate

The final PDFs pass the official ACL checker as short papers:

```bash
uvx --from git+https://github.com/acl-org/aclpubcheck \
  aclpubcheck --paper_type short papers/task1/main.pdf
```

Repeat for `task2` and `task3`.

## Package

```bash
bash papers/package.sh
```

This creates one clean source archive per paper under `papers/dist/`, plus `SHA256SUMS`. For the OpenReview paper submission, upload the corresponding PDF; keep the ZIP as the reproducible source bundle.


The public copies omit the personal contact email from the author block. Scientific content and author affiliations match the accepted manuscripts.
