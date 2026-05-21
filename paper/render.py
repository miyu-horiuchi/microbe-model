"""Render paper/manuscript.md → paper/manuscript.html → paper/manuscript.pdf
in the arXiv PRIME AI style template look (small-caps title with double rules,
centered abstract block with narrower margins, numbered serif section headings,
booktabs tables, running header)."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import markdown


def split_frontmatter(text: str) -> tuple[str, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
    return "", text


def get_field(frontmatter: str, key: str) -> str | None:
    m = re.search(rf'^{re.escape(key)}:\s*"([^"]+)"', frontmatter, re.M)
    return m.group(1) if m else None


def main() -> None:
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/manuscript.md")
    src = src_path.read_text()
    frontmatter, body = split_frontmatter(src)

    title = get_field(frontmatter, "title") or "Manuscript"
    header_text = get_field(frontmatter, "header") or "HMM-Gated PLM Embeddings"
    # Hard-coded author + affiliation block; edit here when collaborators are confirmed.
    author_block_html = """
      <table class="authors"><tr>
        <td><strong>Miyu Horiuchi</strong><br>
            <span class="email">miyuh@alumni.upenn.edu</span></td>
      </tr></table>
    """

    # Extract first paragraph after the # Abstract header to be the styled abstract.
    # The rest of the document (everything from `# 1. Introduction` onward) renders normally.
    abs_match = re.search(r"#\s+Abstract\s*\n+(.+?)(?=\n#\s)", body, re.S)
    abstract_md = abs_match.group(1).strip() if abs_match else ""
    rest_md = body[abs_match.end():] if abs_match else body

    keywords_field = re.findall(r"^\s*-\s+(.+)$", frontmatter[frontmatter.find("keywords:"):], re.M) \
        if "keywords:" in frontmatter else []
    keywords_html = " · ".join(k.strip() for k in keywords_field) if keywords_field else ""

    abstract_html = markdown.markdown(abstract_md, extensions=["tables", "fenced_code"])
    rest_html = markdown.markdown(rest_md, extensions=["tables", "fenced_code"])

    css = r"""
    @page {
      size: letter;
      margin: 1in 1.25in;
      @top-left { content: "__HEADER__"; font-family: 'EB Garamond', 'Georgia', serif; font-size: 9pt; color: #333; }
      @bottom-center { content: counter(page); font-family: 'EB Garamond', 'Georgia', serif; font-size: 10pt; color: #333; }
    }
    body {
      font-family: 'Latin Modern Roman', 'EB Garamond', 'Charter', 'Georgia', serif;
      font-size: 10.5pt;
      line-height: 1.4;
      text-align: justify;
      color: #111;
      max-width: none;
      margin: 0;
      padding: 0;
    }
    h1.paper-title {
      text-align: center;
      text-transform: lowercase;
      font-variant: small-caps;
      font-weight: bold;
      font-size: 18pt;
      letter-spacing: 0.04em;
      border-top: 4px solid #000;
      border-bottom: 1.5px solid #000;
      padding: 12px 0 14px;
      margin: 8px 0 18px;
      line-height: 1.2;
    }
    table.authors {
      margin: 0 auto 18px;
      border-collapse: collapse;
    }
    table.authors td {
      text-align: center;
      padding: 0 32px;
      vertical-align: top;
      line-height: 1.3;
      font-size: 11pt;
    }
    table.authors .email { font-family: 'Menlo', 'Consolas', monospace; font-size: 9pt; }
    .abstract-block {
      margin: 18px 0.5in 18px;
    }
    .abstract-heading {
      text-align: center;
      font-variant: small-caps;
      letter-spacing: 0.05em;
      font-weight: bold;
      font-size: 11pt;
      margin: 6px 0 6px;
    }
    .abstract-block p {
      font-size: 10pt;
      text-align: justify;
      margin: 0;
    }
    .keywords {
      margin: 6px 0 24px;
      font-size: 10pt;
    }
    .keywords strong { font-variant: small-caps; letter-spacing: 0.03em; margin-right: 6px; }
    figure, p > img { display: block; max-width: 92%; height: auto; margin: 14px auto 4px; }
    img + br { display: none; }
    h1, h2, h3 {
      font-family: 'Latin Modern Roman', 'EB Garamond', 'Charter', 'Georgia', serif;
      font-weight: bold;
      color: #000;
      page-break-after: avoid;
    }
    h1 {
      font-size: 13pt;
      margin: 22px 0 8px;
      border: none;
      padding: 0;
    }
    h2 { font-size: 11.5pt; margin: 16px 0 6px; }
    h3 { font-size: 10.5pt; margin: 12px 0 4px; font-style: italic; font-weight: normal; }
    p { margin: 0 0 6px; text-align: justify; }
    table:not(.authors) {
      border-collapse: collapse;
      margin: 12px auto;
      font-size: 9.5pt;
      min-width: 60%;
    }
    table:not(.authors) th {
      border-top: 1.2px solid #000;
      border-bottom: 0.6px solid #000;
      padding: 4px 10px;
      text-align: left;
      font-weight: bold;
      background: none;
    }
    table:not(.authors) td {
      padding: 3px 10px;
      border: none;
    }
    table:not(.authors) tr:last-child td {
      border-bottom: 1.2px solid #000;
    }
    code {
      font-family: 'Menlo', 'Consolas', monospace;
      font-size: 9pt;
      background: #f4f4f4;
      padding: 1px 4px;
      border-radius: 2px;
    }
    pre code {
      display: block;
      padding: 6px 10px;
      overflow-x: auto;
      background: #f4f4f4;
      border-left: 2px solid #888;
    }
    hr { display: none; }
    strong { color: #000; }
    em { color: #222; font-style: italic; }
    ul, ol { margin: 6px 0 6px 22px; padding: 0; }
    li { margin: 2px 0; text-align: justify; }
    """

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;700&display=swap" rel="stylesheet">
<style>{css}</style></head>
<body>
  <h1 class="paper-title">{title}</h1>
  {author_block_html}
  <div class="abstract-block">
    <div class="abstract-heading">Abstract</div>
    {abstract_html}
  </div>
  <p class="keywords"><strong>Keywords</strong>{(" " + keywords_html) if keywords_html else ""}</p>
  {rest_html}
</body></html>"""

    html = html.replace("__HEADER__", header_text)
    html_path = src_path.with_suffix(".html")
    pdf_path = src_path.with_suffix(".pdf")
    html_path.write_text(html)
    print(f"Wrote {html_path}")

    subprocess.run([
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless", "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.absolute()}",
        f"file://{html_path.absolute()}",
    ], check=True)
    print(f"Wrote {pdf_path.absolute()}")


if __name__ == "__main__":
    main()
