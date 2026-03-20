from __future__ import annotations

import html
from typing import Literal

import pandas as pd


def render_html_table(
    title: str,
    note: str,
    frame: pd.DataFrame,
    *,
    escape: bool = True,
    variant: Literal["standard", "compact"] = "standard",
) -> str:
    safe_note = note if not escape else html.escape(note)
    if variant == "compact":
        table_html = frame.to_html(index=False, border=0, escape=escape, classes="dataframe")
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<style>"
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:2rem;}"
            "table{border-collapse:collapse;width:100%;}"
            "th,td{border:1px solid #ddd;padding:0.5rem;text-align:left;font-size:0.9rem;}"
            "th{background:#f6f8fa;}"
            "</style></head><body>"
            f"<h1>{title}</h1><p>{safe_note}</p>{table_html}</body></html>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f5f5f5; }}
    .note {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin-bottom: 12px; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="note">{safe_note}</div>
  {frame.to_html(index=False, escape=escape)}
</body>
</html>
"""
