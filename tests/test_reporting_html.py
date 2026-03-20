from __future__ import annotations

import pandas as pd

from mekubbal.reporting.html import render_html_table


def test_render_html_table_standard_includes_title_and_note():
    html = render_html_table("Example", "A note", pd.DataFrame([{"a": 1}]))

    assert "<h1>Example</h1>" in html
    assert "A note" in html


def test_render_html_table_compact_allows_unescaped_cells():
    html = render_html_table(
        "Compact",
        "x < y",
        pd.DataFrame([{"a": "<strong>ok</strong>"}]),
        escape=False,
        variant="compact",
    )

    assert "<p>x < y</p>" in html
    assert "<strong>ok</strong>" in html
