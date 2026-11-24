"""PDF Layout Test: the compiler must write a valid PDF and never raise styling errors."""
import pandas as pd
import pytest
from pypdf import PdfReader

from crew.report_generator import (
    ReportGenerationError,
    generate_pdf_report,
    inline_markup,
    sanitize,
)
from crew.tools.financial_tool import compute_summary_df, load_metrics

SAMPLE_MD = """# Title that becomes the banner
## Executive Summary
Revenue grew **strongly** in *Q3*. Margins & growth stayed <healthy>.

## Key Financial Metrics
| metric | mean | max |
| --- | --- | --- |
| revenue | 25.00 | 40.00 |

### Notes
- Bullet one
- Bullet two
1. Numbered item
"""


@pytest.fixture()
def frames(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("company,revenue,profit\nA,10,2\nB,20,4\nC,30,6\n")
    raw = load_metrics(csv)
    return raw, compute_summary_df(raw)


def test_writes_valid_pdf_file(tmp_path, frames):
    raw, summary = frames
    out = generate_pdf_report(SAMPLE_MD, summary, tmp_path / "market_report.pdf",
                              metadata={"query": "Tech Sector Q3 Analysis", "run_id": "t1"}, raw_df=raw)
    assert out.exists() and out.stat().st_size > 1500
    assert out.read_bytes().startswith(b"%PDF")
    reader = PdfReader(str(out))
    assert len(reader.pages) >= 2            # report page(s) + appendix page
    assert reader.pages[0].mediabox.width == pytest.approx(595.27, abs=1)   # A4


def test_pdf_contains_banner_tables_and_page_numbers(tmp_path, frames):
    raw, summary = frames
    out = generate_pdf_report(SAMPLE_MD, summary, tmp_path / "r.pdf",
                              metadata={"query": "Tech Sector Q3 Analysis"}, raw_df=raw)
    reader = PdfReader(str(out))
    total = len(reader.pages)
    first = reader.pages[0].extract_text()
    assert "Market Research Report" in first and "Tech Sector Q3 Analysis" in first   # banner
    assert "Author:" not in first
    assert "Executive Summary" in first
    full = "\n".join(p.extract_text() for p in reader.pages)
    assert "Quantitative Summary" in full and "25.00" in full                         # pandas table
    assert "For informational purposes only." in full
    for n, page in enumerate(reader.pages, start=1):
        assert f"Page {n} of {total}" in page.extract_text()


def test_long_report_flows_over_multiple_pages(tmp_path, frames):
    raw, summary = frames
    long_md = "## Section\n" + "\n\n".join(f"Paragraph {i}. " + "word " * 80 for i in range(40))
    out = generate_pdf_report(long_md, summary, tmp_path / "long.pdf", raw_df=raw)
    reader = PdfReader(str(out))
    assert len(reader.pages) >= 4
    assert f"Page {len(reader.pages)} of {len(reader.pages)}" in reader.pages[-1].extract_text()


@pytest.mark.parametrize("nasty", [
    "",                                                     # empty writer output
    "## H\nText with unicode -> \u2248 \u2265 \u2713 \u4e2d\u6587 emoji \U0001F680",
    "## H\n<b>unclosed <i> tags & ampersands &amp; </para>",
    "| a | b |\n| --- | --- |\n| only one cell |",           # ragged table
    "## H\n```python\nprint('x')\n```",
])
def test_awkward_markdown_never_raises(tmp_path, frames, nasty):
    raw, summary = frames
    out = generate_pdf_report(nasty, summary, tmp_path / "n.pdf", raw_df=raw)
    assert out.read_bytes().startswith(b"%PDF")


def test_works_without_raw_dataset(tmp_path, frames):
    _, summary = frames
    out = generate_pdf_report(SAMPLE_MD, summary, tmp_path / "x.pdf")
    assert out.exists()


def test_unwritable_target_raises_report_generation_error(tmp_path, frames):
    raw, summary = frames
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(ReportGenerationError):
        generate_pdf_report(SAMPLE_MD, summary, blocker / "sub" / "r.pdf", raw_df=raw)


def test_inline_markup_and_sanitize():
    assert inline_markup("**b** & <x>") == "<b>b</b> &amp; &lt;x&gt;"
    assert sanitize("a \u2192 b") == "a -> b"
    assert isinstance(sanitize("\U0001F680"), str)


def test_wrapped_bullet_continuation_stays_in_same_list_item():
    from crew.report_generator import build_styles, markdown_to_flowables

    md = "- **Spread:** margin spans -6% to\n  31%, so quality differs.\n- Second item\n"
    flow = markdown_to_flowables(md, build_styles())
    assert len(flow) == 2                       # two bullets, no stray paragraph
    assert "31%, so quality differs" in flow[0].text
