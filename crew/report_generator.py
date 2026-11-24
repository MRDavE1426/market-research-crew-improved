"""Programmatic A4 PDF report compiler (ReportLab / Platypus).

Input : the Content Writer's Markdown + the Pandas summary DataFrame.
Output: a typeset PDF with
  * a header banner carrying document metadata (first page: full banner,
    following pages: slim running header),
  * styled sections rendered from the writer's Markdown (headings, paragraphs,
    bullets, tables),
  * a tabular grid with the Pandas summary statistics,
  * automatic "Page X of Y" numbering and a footer note on every page.
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .tools.financial_tool import _fmt, compute_summary_df, data_quality_checks

PAGE_W, PAGE_H = A4
MARGIN_X = 2.0 * cm
TOP_MARGIN = 3.7 * cm
BOTTOM_MARGIN = 2.3 * cm
FIRST_BANNER_H = 3.1 * cm
LATER_BANNER_H = 1.5 * cm
CONTENT_W = PAGE_W - 2 * MARGIN_X

NAVY = colors.HexColor("#0B2545")
BLUE = colors.HexColor("#13589B")
LIGHT = colors.HexColor("#EEF3F9")
GRID = colors.HexColor("#B8C4D6")
GREY = colors.HexColor("#5C6B7F")

DEFAULT_FOOTER = "For informational purposes only."


class ReportGenerationError(RuntimeError):
    """Raised when the PDF cannot be compiled."""


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
_CHAR_MAP = {
    "\u2192": "->", "\u2190": "<-", "\u2248": "~", "\u2265": ">=", "\u2264": "<=",
    "\u2212": "-", "\u2011": "-", "\u2010": "-", "\u00a0": " ", "\u200b": "",
    "\u2713": "v", "\u2714": "v", "\u2022": "\u2022", "\u25cf": "\u2022",
}


def sanitize(text: str) -> str:
    """Standard PDF fonts only cover Latin-1/WinAnsi; map or drop everything else."""
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def inline_markup(text: str) -> str:
    """Escape text for ReportLab Paragraphs and convert inline Markdown."""
    text = html.escape(sanitize(text), quote=False)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    return text


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------
def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=10,
                               leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                             leading=20, textColor=NAVY, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13.5,
                             leading=17, textColor=BLUE, spaceBefore=14, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11,
                             leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=3),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=10,
                                 leading=14, leftIndent=16, bulletIndent=4, spaceAfter=2),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=11, alignment=TA_CENTER),
        "cell_left": ParagraphStyle("cell_left", fontName="Helvetica", fontSize=8.6, leading=11),
        "cell_head": ParagraphStyle("cell_head", fontName="Helvetica-Bold", fontSize=8.6, leading=11,
                                    textColor=colors.white, alignment=TA_CENTER),
        "warning": ParagraphStyle("warning", fontName="Helvetica-Bold", fontSize=9, leading=12,
                                  textColor=colors.HexColor("#7B241C")),
        "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
                                  textColor=GREY, spaceBefore=3, spaceAfter=10),
    }


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def build_table(header: list[str], rows: list[list[str]], styles: dict, col_widths=None) -> Table:
    ncols = len(header)
    data = [[Paragraph(inline_markup(h), styles["cell_head"]) for h in header]]
    for row in rows:
        row = (row + [""] * ncols)[:ncols]
        data.append(
            [Paragraph(inline_markup(c), styles["cell_left"] if i == 0 else styles["cell"])
             for i, c in enumerate(row)]
        )
    if col_widths is None:
        first = min(CONTENT_W * 0.30, CONTENT_W / ncols * 1.5) if ncols > 1 else CONTENT_W
        rest = (CONTENT_W - first) / max(ncols - 1, 1)
        col_widths = [first] + [rest] * (ncols - 1)
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    table.setStyle(TableStyle(style))
    return table


def dataframe_table(df: pd.DataFrame, styles: dict, index: bool = True, decimals: int = 2) -> Table:
    frame = df.reset_index() if index else df
    header = [str(c).replace("_", " ") for c in frame.columns]
    rows = [[_fmt(v, decimals) for v in row] for row in frame.itertuples(index=False, name=None)]
    return build_table(header, rows, styles)


def build_warning_box(message: str, styles: dict) -> Table:
    """Full-width red notice used when the report is not based on real researched data."""
    box = Table([[Paragraph(inline_markup(message), styles["warning"])]], colWidths=[CONTENT_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FDECEA")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#C0392B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return box


# --------------------------------------------------------------------------
# Markdown -> flowables
# --------------------------------------------------------------------------
_TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def markdown_to_flowables(md: str, styles: dict, skip_first_h1: bool = True) -> list:
    flow: list = []
    lines = md.replace("\r\n", "\n").split("\n")
    para: list[str] = []
    bullet: list = []          # [bullet_text, [text lines]] while a list item is open
    seen_h1 = False
    i = 0

    def flush_para():
        if bullet:
            flow.append(Paragraph(inline_markup(" ".join(bullet[1])), styles["bullet"], bulletText=bullet[0]))
            bullet.clear()
        if para:
            flow.append(Paragraph(inline_markup(" ".join(para)), styles["body"]))
            para.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush_para()
            i += 1
            continue
        if line.startswith("```"):  # skip fenced code delimiters, keep content as text
            flush_para()
            i += 1
            continue
        # table block
        if line.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1].strip()):
            flush_para()
            header = _split_row(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            flow.append(build_table(header, rows, styles))
            flow.append(Spacer(1, 8))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para()
            level, title = len(m.group(1)), m.group(2).strip().strip("#").strip()
            if level == 1 and skip_first_h1 and not seen_h1:
                seen_h1 = True
            else:
                style = styles["h1"] if level == 1 else styles["h2"] if level == 2 else styles["h3"]
                flow.append(Paragraph(inline_markup(title), style))
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", line):
            flush_para()
            i += 1
            continue
        b = re.match(r"^([-*+])\s+(.*)$", line)
        n = re.match(r"^(\d+)[.)]\s+(.*)$", line)
        if b or n:
            flush_para()
            bullet[:] = ["\u2022" if b else f"{n.group(1)}.", [(b or n).group(2)]]
            i += 1
            continue
        if bullet and raw[:1] in (" ", "\t"):   # wrapped continuation of a list item
            bullet[1].append(line)
            i += 1
            continue
        if line.startswith(">"):
            line = line.lstrip("> ").strip()
        if bullet:
            flush_para()
        para.append(line)
        i += 1
    flush_para()
    return flow


# --------------------------------------------------------------------------
# Canvas: "Page X of Y" + footer note
# --------------------------------------------------------------------------
def make_numbered_canvas(footer_note: str):
    class NumberedCanvas(rl_canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states: list[dict] = []

        def showPage(self):  # noqa: N802 (ReportLab API)
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_footer(total)
                super().showPage()
            super().save()

        def _draw_footer(self, total: int):
            self.saveState()
            self.setStrokeColor(GRID)
            self.setLineWidth(0.6)
            self.line(MARGIN_X, 1.65 * cm, PAGE_W - MARGIN_X, 1.65 * cm)
            self.setFont("Helvetica", 7.5)
            self.setFillColor(GREY)
            self.drawString(MARGIN_X, 1.2 * cm, sanitize(footer_note)[:120])
            self.setFont("Helvetica-Bold", 8.5)
            self.setFillColor(NAVY)
            self.drawRightString(PAGE_W - MARGIN_X, 1.2 * cm, f"Page {self._pageNumber} of {total}")
            self.restoreState()

    return NumberedCanvas


def _draw_banner(canv, doc, meta: dict, first: bool):
    canv.saveState()
    height = FIRST_BANNER_H if first else LATER_BANNER_H
    canv.setFillColor(NAVY)
    canv.rect(0, PAGE_H - height, PAGE_W, height, stroke=0, fill=1)
    canv.setFillColor(BLUE)
    canv.rect(0, PAGE_H - height - 0.12 * cm, PAGE_W, 0.12 * cm, stroke=0, fill=1)
    canv.setFillColor(colors.white)
    if first:
        canv.setFont("Helvetica-Bold", 20)
        canv.drawString(MARGIN_X, PAGE_H - 1.35 * cm, sanitize(meta["title"])[:60])
        canv.setFont("Helvetica", 10.5)
        canv.drawString(MARGIN_X, PAGE_H - 1.95 * cm, sanitize(f"Topic: {meta['query']}")[:95])
        canv.setFont("Helvetica", 8.5)
        canv.setFillColor(colors.HexColor("#C9D6EA"))
        line1 = f"Generated: {meta['generated']}   |   Run ID: {meta['run_id']}   |   Mode: {meta['mode']}"
        line2 = f"Data source: {meta['data_source']}"
        canv.drawString(MARGIN_X, PAGE_H - 2.5 * cm, sanitize(line1)[:130])
        canv.drawString(MARGIN_X, PAGE_H - 2.9 * cm, sanitize(line2)[:130])
    else:
        canv.setFont("Helvetica-Bold", 9.5)
        canv.drawString(MARGIN_X, PAGE_H - 0.9 * cm, sanitize(meta["title"])[:60])
        canv.setFont("Helvetica", 8.5)
        canv.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 0.9 * cm, sanitize(meta["query"])[:55])
    canv.restoreState()


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def generate_pdf_report(
    markdown_text: str,
    summary_df: pd.DataFrame,
    output_path: str | Path,
    metadata: dict | None = None,
    raw_df: pd.DataFrame | None = None,
    footer_note: str = DEFAULT_FOOTER,
) -> Path:
    """Compile the A4 PDF. Raises ``ReportGenerationError`` on any failure."""
    output_path = Path(output_path)
    meta = {
        "title": "Market Research Report",
        "query": "N/A",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_id": "-",
        "mode": "live",
        "author": "",
        "data_source": "-",
    }
    meta.update(metadata or {})
    styles = build_styles()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title=sanitize(f"{meta['title']} - {meta['query']}"),
            author=sanitize(meta["author"]),
            subject=sanitize(f"Market research on: {meta['query']}"),
            creator="market-research-crew",
        )

        story: list = []
        if str(meta["data_source"]).upper().startswith("FALLBACK"):
            story.append(build_warning_box(
                "DATA WARNING: the research agent returned no usable metrics, so the tables in this "
                "report come from a bundled SYNTHETIC sample dataset. Do not use these figures for decisions.",
                styles))
        story += markdown_to_flowables(markdown_text or "", styles)
        if not story:
            story.append(Paragraph("The writer agent returned no report text.", styles["body"]))

        story.append(Paragraph("Quantitative Summary (Pandas)", styles["h2"]))
        story.append(
            Paragraph(
                "Descriptive statistics computed with Pandas over every numeric metric "
                "collected by the research agent.",
                styles["body"],
            )
        )
        story.append(KeepTogether([dataframe_table(summary_df, styles)]))
        story.append(Paragraph("Table 1 - count, mean, standard deviation, minimum and maximum per metric.",
                               styles["caption"]))

        # Appendix always starts on a fresh page -> report is multi-page.
        story.append(PageBreak())
        story.append(Paragraph("Appendix A - Dataset & Data Quality", styles["h2"]))
        if raw_df is not None and not raw_df.empty:
            shown = raw_df.head(30)
            story.append(Paragraph("Collected dataset", styles["h3"]))
            story.append(dataframe_table(shown, styles, index=False))
            if len(raw_df) > len(shown):
                story.append(Paragraph(f"Showing first {len(shown)} of {len(raw_df)} rows.", styles["caption"]))
            story.append(Spacer(1, 6))
            story.append(Paragraph("Numeric consistency checks", styles["h3"]))
            for finding in data_quality_checks(raw_df):
                story.append(Paragraph(inline_markup(finding), styles["bullet"], bulletText="\u2022"))
        else:
            story.append(Paragraph("No raw dataset was supplied.", styles["body"]))

        doc.build(
            story,
            onFirstPage=lambda c, d: _draw_banner(c, d, meta, first=True),
            onLaterPages=lambda c, d: _draw_banner(c, d, meta, first=False),
            canvasmaker=make_numbered_canvas(footer_note),
        )
    except Exception as exc:
        raise ReportGenerationError(f"PDF generation failed: {exc}") from exc

    return output_path


def generate_pdf_from_files(markdown_path, csv_path, output_path, metadata=None) -> Path:
    """Convenience wrapper used by the pipeline."""
    from .tools.financial_tool import load_metrics

    raw = load_metrics(csv_path)
    return generate_pdf_report(
        Path(markdown_path).read_text(encoding="utf-8"),
        compute_summary_df(raw),
        output_path,
        metadata=metadata,
        raw_df=raw,
    )
