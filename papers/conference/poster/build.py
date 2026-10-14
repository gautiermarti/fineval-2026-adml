#!/usr/bin/env python3
"""Build an editable vector SVG and an A0 print PDF from content.json.

Requires Python 3, reportlab, and DejaVu Sans (regular and bold).
All coordinates use points with a top-left origin. The same drawing operations
produce both outputs, keeping the editable SVG aligned with the print PDF.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.graphics.barcode.qrencoder import QRCode, QRErrorCorrectLevel
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent
W, H = 841 * 72 / 25.4, 1189 * 72 / 25.4
M = 72
IW = W - 2 * M
GAP = 32
CW = (IW - GAP) / 2
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
REGULAR = FONT_DIR / 'DejaVuSans.ttf'
BOLD = FONT_DIR / 'DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('Poster', str(REGULAR)))
pdfmetrics.registerFont(TTFont('PosterBold', str(BOLD)))

INK = '#152A3B'
MUTED = '#506371'
TEAL = '#087F83'
MINT = '#E9F5F1'
BLUE = '#ECF2F8'
LIGHT = '#F5F7F9'
LINE = '#D6E0E5'
AMBER = '#AA5C20'
AMBER_LIGHT = '#FFF2E5'
WHITE = '#FFFFFF'


class Drawing:
    def __init__(self, logo_filename=False):
        stem = 'FinNLP65ADMLPosterLogo' if logo_filename else 'FinNLP65ADMLPoster'
        self.pdf_path = OUT / (stem + '.pdf')
        self.svg_path = OUT / (stem + '.svg')
        self.pdf = Canvas(str(self.pdf_path), pagesize=(W, H), pageCompression=1,
                          initialFontName='Poster', initialFontSize=24, invariant=1)
        self.pdf.setTitle('ADML at the FinEval 2026: Version-Aware Retrieval and Grounded Sparse Editing for Multilingual Financial QA')
        self.pdf.setAuthor('Gautier Marti; Khaled AlNuaimi')
        self.pdf.setSubject('Agent-led research across three financial NLP tasks; FinNLP 2026')
        fonts = '\n'.join(
            '@font-face{font-family:Poster; font-weight:%s; src:url(data:font/ttf;base64,%s)}'
            % (weight, base64.b64encode(path.read_bytes()).decode())
            for weight, path in [(400, REGULAR), (700, BOLD)]
        )
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="841mm" height="1189mm" viewBox="0 0 {W:.5f} {H:.5f}">',
                    '<title>ADML FinEval 2026 poster</title>', f'<defs><style>{fonts}</style></defs>']
        self.text_bounds = []
        self.texts = []
        self.image_bounds = []
        self.qr_info = None

    def qr_code(self, x, y, size, url):
        qr = QRCode(None, QRErrorCorrectLevel.M)
        qr.addData(url)
        qr.make()
        count, border = qr.getModuleCount(), 4
        unit = size / (count + 2*border)
        self.svg.append(f'<a href="{escape(url)}">')
        self.rect(x, y, size, size, WHITE)
        for row in range(count):
            for column in range(count):
                if qr.isDark(row, column):
                    self.rect(x+(column+border)*unit, y+(row+border)*unit,
                              unit, unit, '#000000')
        self.svg.append('</a>')
        self.pdf.linkURL(url, (x, H-y-size, x+size, H-y), relative=0, thickness=0)
        self.image_bounds.append([x, y, x+size, y+size])
        self.qr_info = {'url': url, 'size_mm': size*25.4/72, 'modules': count,
                        'quiet_zone_modules': border, 'error_correction': 'M',
                        'rendering': 'vector', 'bounds_points': [x, y, x+size, y+size]}

    def image(self, path, x, y, width):
        reader = ImageReader(str(path))
        source_width, source_height = reader.getSize()
        height = width * source_height / source_width
        self.pdf.drawImage(reader, x, H-y-height, width, height, mask='auto')
        encoded = base64.b64encode(path.read_bytes()).decode()
        self.svg.append(f'<image x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>')
        self.image_bounds.append([x, y, x+width, y+height])

    def rect(self, x, y, w, h, fill, stroke=None, radius=0):
        self.pdf.setFillColor(HexColor(fill))
        self.pdf.setStrokeColor(HexColor(stroke or fill))
        if radius:
            self.pdf.roundRect(x, H-y-h, w, h, radius, fill=1, stroke=bool(stroke))
        else:
            self.pdf.rect(x, H-y-h, w, h, fill=1, stroke=bool(stroke))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke or "none"}"/>')

    def line(self, x1, y1, x2, y2, color=LINE, width=1.5):
        self.pdf.setStrokeColor(HexColor(color))
        self.pdf.setLineWidth(width)
        self.pdf.line(x1, H-y1, x2, H-y2)
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')

    def text(self, x, y, text, size=25, bold=False, color=INK, align='left'):
        font = 'PosterBold' if bold else 'Poster'
        tw = pdfmetrics.stringWidth(text, font, size)
        x0 = x - tw if align == 'right' else x-tw/2 if align == 'center' else x
        baseline = y + pdfmetrics.getAscent(font, size)
        height = pdfmetrics.getAscent(font, size) - pdfmetrics.getDescent(font, size)
        self.pdf.setFont(font, size)
        self.pdf.setFillColor(HexColor(color))
        self.pdf.drawString(x0, H-baseline, text)
        self.svg.append(f'<text x="{x0}" y="{baseline}" font-family="Poster, DejaVu Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(text)}</text>')
        self.text_bounds.append([x0, y, x0+tw, y+height])
        self.texts.append(text)

    def wrap(self, text, width, size, bold=False):
        font = 'PosterBold' if bold else 'Poster'
        lines = []
        for paragraph in text.split('\n'):
            line = ''
            for word in paragraph.split():
                candidate = f'{line} {word}'.strip()
                if pdfmetrics.stringWidth(candidate, font, size) <= width:
                    line = candidate
                else:
                    if not line:
                        raise ValueError(f'Word does not fit: {word}')
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines

    def paragraph(self, x, y, text, width, size=25, bold=False, color=INK, leading=None, max_height=None):
        leading = leading or size * 1.30
        lines = self.wrap(text, width, size, bold)
        height = len(lines) * leading
        if max_height is not None and height > max_height + 0.01:
            raise ValueError(f'Paragraph overflow ({height:.1f}>{max_height}): {text}')
        for i, line in enumerate(lines):
            self.text(x, y+i*leading, line, size, bold, color)
        return y + height

    def arrow(self, x1, y, x2, color=TEAL):
        self.line(x1, y, x2, y, color, 2.5)
        self.line(x2-9, y-7, x2, y, color, 2.5)
        self.line(x2-9, y+7, x2, y, color, 2.5)

    def section(self, x, y, number, title, size=35):
        self.text(x, y+3, number, 24, True, TEAL)
        self.text(x+59, y, title, size, True)

    def save(self):
        minimum = 10 * 72 / 25.4
        outside = [b for b in self.text_bounds if min(b[0], b[1], W-b[2], H-b[3]) < minimum]
        if outside:
            raise ValueError(f'Text outside 10 mm safe area: {outside}')
        if any(min(b[0], b[1], W-b[2], H-b[3]) < minimum for b in self.image_bounds):
            raise ValueError('Graphic outside 10 mm safe area')
        self.pdf.showPage()
        self.pdf.save()
        self.svg.append('</svg>')
        self.svg_path.write_text('\n'.join(self.svg))
        (HERE/'poster_text.txt').write_text('\n'.join(self.texts)+'\n')
        return {
            'page_mm': [841, 1189],
            'page_points': [W, H],
            'orientation': 'portrait',
            'page_count': 1,
            'all_drawn_text_inside_10mm_safe_area': True,
            **({'all_images_inside_10mm_safe_area': True} if self.image_bounds else {}),
            'smallest_text_pt': 17,
            'main_body_text_pt': '24–26',
            'graphics': 'vector text, shapes and charts; original official EMNLP logo PNG',
            **({'qr_code': self.qr_info} if self.qr_info else {}),
            'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in [self.pdf_path, self.svg_path]},
        }


def repo_counts():
    return json.loads((HERE/'research_inventory.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conference-logo', action='store_true',
                        help='Use the earlier PosterLogo filenames for the same approved design')
    args = parser.parse_args()
    c = json.loads((HERE/'content.json').read_text())
    counts = repo_counts()
    d = Drawing(logo_filename=args.conference_logo)
    d.rect(0, 0, W, H, WHITE)
    d.text(M, 69, 'FinNLP 2026  /  EMNLP  /  Budapest  /  28 October 2026', 22, True, TEAL)
    d.text(W-M, 69, 'TEAM ADML', 22, True, INK, 'right')
    for i, title in enumerate(c['title_lines']):
        d.text(M, 115+66*i, title, 55, True)
    d.text(M, 335, c['authors'], 31, True)
    d.text(M, 380, c['affiliations'], 25, color=MUTED)
    d.text(M, 434, c['subtitle'], 31, color=TEAL)
    d.image(HERE/'assets'/'emnlp_2026_logo_v1.png', W-M-600, 340, 600)
    d.line(M, 489, W-M, 489)

    # Official private results. Scores use different task metrics.
    rw = (IW-2*GAP)/3
    for i, result in enumerate(c['results']):
        x, y = M+i*(rw+GAP), 514
        d.rect(x, y, rw, 158, INK, radius=12)
        d.text(x+25, y+21, result['task'], 24, True, WHITE)
        d.text(x+25, y+66, result['rank'], 51, True, WHITE)
        d.text(x+rw-25, y+83, result['metric'], 23, color='#CFDFE7', align='right')
        d.text(x+25, y+127, 'OFFICIAL PRIVATE RESULT', 17, True, '#AAC9D7')

    d.rect(M, 698, IW, 123, MINT, radius=12)
    d.text(M+27, 720, c['thesis_title'], 36, True, TEAL)
    d.text(M+27, 775, c['thesis_body'], 25)

    d.section(M, 854, '01', 'The research process')
    d.text(M, 911, 'An agent-led workflow implemented experiments, checked evidence and recorded failed hypotheses.', 27)
    bw = (IW-4*35)/5
    for i, (title, subtitle) in enumerate(c['loop']):
        x, y = M+i*(bw+35), 963
        d.rect(x, y, bw, 111, LIGHT, LINE, 8)
        d.text(x+19, y+20, title, 28, True)
        d.text(x+19, y+68, subtitle, 22, color=MUTED)
        if i < 4:
            d.arrow(x+bw+5, y+55, x+bw+29)
    d.text(M, 1094, 'Promotion required independent evidence: entity/source/company groups, seed checks, harm counts and temporal confirmation.', 24, color=MUTED)
    stats = [
        (str(counts['scripts']['files']), 'research scripts'),
        (str(counts['src']['files']), 'Python modules'),
        (str(counts['tests']['files']), 'test files'),
        (str(counts['preregistration_documents']), 'preregistration documents'),
    ]
    for i, (value, label) in enumerate(stats):
        x = M+i*IW/4
        d.text(x, 1144, value, 40, True, TEAL)
        d.text(x+105, 1157, label, 25)
    d.text(M, 1206, 'Repository snapshot: 340 Python files; 48,468 physical lines including tests. These measure code scope, not experiment count or productivity.', 20, color=MUTED)

    d.section(M, 1262, '02', 'Three systems developed for new inputs')
    for i, card in enumerate(c['research_cards']):
        x, y = M+i*(rw+GAP), 1325
        d.rect(x, y, rw, 382, LIGHT, LINE, 10)
        d.text(x+24, y+22, card['tag'], 20, True, TEAL)
        d.paragraph(x+24, y+62, card['title'], rw-48, 27, True, max_height=74)
        d.paragraph(x+24, y+138, card['models'], rw-48, 24, True, max_height=64)
        d.paragraph(x+24, y+213, card['methods'], rw-48, 23, max_height=94)
        d.paragraph(x+24, y+310, card['validation'], rw-48, 22, color=MUTED, max_height=65)

    # QA is the main paper's case study, using only historical measurements.
    x, y = M, 1756
    d.section(x, y, '03', 'QA: candidate quality versus selection', 32)
    d.text(x, y+60, 'Grounded compiler: every number has a source or derivation.', 25)
    qw = (CW-3*24)/4
    for i, (title, detail) in enumerate(c['qa_compiler']):
        bx = x+i*(qw+24)
        d.rect(bx, y+112, qw, 112, MINT, radius=8)
        d.text(bx+14, y+133, title, 25, True, TEAL)
        d.paragraph(bx+14, y+176, detail, qw-28, 21, max_height=55)
        if i < 3:
            d.arrow(bx+qw+3, y+165, bx+qw+20)
    d.text(x, y+259, 'Historical QA · 192 rows · macro ROUGE-1 F1', 23, True, MUTED)
    plotx, plotw, top = x+345, CW-450, y+318
    for i, item in enumerate(c['qa_scores']):
        by = top+i*75
        d.text(x, by+10, item['label'], 24, bold=item['kind']=='oracle')
        color = TEAL if item['kind']=='oracle' else INK if item['kind']=='selected' else '#8CA2B1'
        d.rect(plotx, by, item['score']/0.45*plotw, 43, color, radius=4)
        d.text(plotx+item['score']/0.45*plotw+14, by+6, f"{item['score']:.4f}", 26, True, color)
    axis_y = top+225
    d.line(plotx, axis_y, plotx+plotw, axis_y)
    for tick in [0, .1, .2, .3, .4]:
        tx = plotx+tick/.45*plotw
        d.line(tx, axis_y, tx, axis_y+6)
        d.text(tx, axis_y+15, f'{tick:.1f}', 19, color=MUTED, align='center')
    d.paragraph(x, y+600, 'Oracle gain: +0.0377, using reference answers to choose the edit. This is candidate headroom, not a deployable score.', CW, 25, max_height=100)
    d.rect(x, y+702, CW, 111, MINT, radius=8)
    d.paragraph(x+20, y+720, 'The nested selector abstained on every row. Under both time and company shift, another selector changed 4 rows and harmed all 4.', CW-40, 25, color=TEAL, max_height=84)

    # Concrete candidate decisions; task metrics are intentionally labeled.
    gx = M+CW+GAP
    d.section(gx, y, '04', 'The gates rejected plausible gains', 32)
    d.text(gx, y+60, 'Selected examples; results use different tasks and evaluation sets.', 23, color=MUTED)
    for i, gate in enumerate(c['gates']):
        gy = y+110+i*157
        keep = gate['status']=='KEEP'
        tint, accent = (MINT, TEAL) if keep else (AMBER_LIGHT, AMBER)
        d.rect(gx, gy, CW, 145, LIGHT, radius=8)
        d.text(gx+18, gy+15, gate['title'], 25, True)
        d.rect(gx+CW-144, gy+12, 126, 34, tint, radius=6)
        d.text(gx+CW-81, gy+18, gate['status'], 20, True, accent, 'center')
        d.text(gx+18, gy+55, gate['score'], 30, True, accent)
        d.paragraph(gx+335, gy+56, gate['context'], CW-359, 23, max_height=84)
    d.paragraph(gx, y+756, 'NER validation was repeatedly inspected; the retained gain is development evidence, not a fresh hidden-test result.', CW, 21, color=MUTED, max_height=60)

    d.section(M, 2610, '05', 'Test release changed what the benchmark rewarded')
    d.text(M, 2670, '11 August: test inputs audited.  17 August: external-data permission and public scores resolved the final choices.', 25)
    for i, card in enumerate(c['final_cards']):
        fx, fy = M+i*(rw+GAP), 2725
        d.rect(fx, fy, rw, 229, BLUE, radius=10)
        d.text(fx+24, fy+22, card['title'], 27, True)
        d.text(fx+24, fy+73, card['stat'], 45, True, TEAL)
        d.paragraph(fx+24, fy+139, card['detail'], rw-48, 23, max_height=62)
        d.text(fx+24, fy+201, card['note'], 21, color=MUTED)
    d.text(M, 2980, 'External public data were permitted. Selected inference was CPU-only, with no model or API calls.', 25, color=MUTED)

    d.rect(M, 3040, IW, 173, INK, radius=12)
    d.text(M+26, 3067, c['closing_title'], 33, True, WHITE)
    for i, line in enumerate(c['closing_lines']):
        d.text(M+26, 3124+i*39, line, 25, color='#D7E6EE')

    d.qr_code(M, 3230, 102, c['materials_url'])
    d.text(M+135, 3238, 'Benchmarks: Plutus (EMNLP 2025); Ebisu (2026); MultiFinBen (ACL 2026). Methods/results: ADML FinEval system papers and experiment records.', 19, color=MUTED)
    d.text(M+135, 3280, 'Papers, code & materials', 23, True, TEAL)
    d.text(M+135, 3311, c['materials_url'].removeprefix('https://'), 19, color=MUTED)
    d.text(W-M, 3303, f"{c['paper_id']}  ·  FinNLP 2026  ·  ADML", 24, color=MUTED, align='right')

    audit = d.save()
    audit['repository_counts'] = counts
    audit['approval_status'] = 'See poster/README.md for approval and delivery status'
    checks_name = 'logo_build_checks.json' if args.conference_logo else 'build_checks.json'
    (HERE/checks_name).write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
