# Conference materials

- [A0 portrait poster](FinNLP65ADMLPoster.pdf), [preview](FinNLP65ADMLPosterPreview.png), and [editable SVG](FinNLP65ADMLPoster.svg).
- [Six-slide PDF](FinNLP65ADMLSlides.pdf), [editable PowerPoint](FinNLP65ADMLSlides.pptx), and [overview](FinNLP65ADMLSlidesPreview.jpg).
- [Presentation video](FinNLP65ADMLVideo.mp4), [WebVTT captions](FinNLP65ADMLVideo.vtt), [SRT captions](FinNLP65ADMLVideo.srt), and [transcript](video/script.md).

The video is 4 minutes 49.833 seconds and uses synthetic narration. The poster is one A0 portrait page, 841 × 1189 mm, with embedded fonts. The accepted multilingual QA paper title is retained while the content discusses research across all three tasks.

Rebuild the poster with `python papers/conference/poster/build.py` using ReportLab and the DejaVu Sans fonts under `/usr/share/fonts/truetype/dejavu`. Its research inventory is stored locally, so no private Git history is required. Rebuild slides with `python papers/conference/video/build_slides.py` using ReportLab, python-pptx, Pillow, and Poppler's `pdftoppm`. The delivered MP4 is included; raw voice-cloning inputs are not part of this release.
