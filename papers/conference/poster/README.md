# Poster source

The [print PDF](../FinNLP65ADMLPoster.pdf) is A0 portrait (841 × 1189 mm). [content.json](content.json) contains the text and results; [build.py](build.py) generates the PDF and editable SVG. All text and graphic placements stay inside the 10 mm safe area. The original EMNLP logo is embedded at approximately 1044 dpi; text and diagrams are vector.

```bash
python papers/conference/poster/build.py
pdftoppm -scale-to 2200 -singlefile -png \
  papers/conference/FinNLP65ADMLPoster.pdf \
  papers/conference/FinNLP65ADMLPosterPreview
```

Requirements: Python 3, ReportLab, DejaVu Sans regular/bold fonts and Poppler for the preview. [research_inventory.json](research_inventory.json) preserves the historical research counts independently of the public repository's Git history.

[Claim sources](sources.md) · [Font license](LICENSE-DejaVu.txt) · [Logo provenance](assets/provenance.json)
