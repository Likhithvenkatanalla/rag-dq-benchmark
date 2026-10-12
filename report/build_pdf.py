"""Build report/REPORT.pdf from report/REPORT.md (Markdown -> HTML -> PDF with headless Chromium).

Usage:
    pip install markdown playwright
    python report/build_pdf.py [--chromium /path/to/chrome]
"""
import argparse
import os

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10.2pt; line-height: 1.42; color: #111; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 6pt 0; }
h1 + p { margin-top: 0; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt 0; border-bottom: 0.6pt solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 10.8pt; margin: 11pt 0 4pt 0; }
p, li { text-align: justify; hyphens: auto; }
ul { padding-left: 16pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 2pt 0; font-size: 8.6pt;
        font-family: "DejaVu Sans", Arial, sans-serif; page-break-inside: avoid; }
th, td { border-bottom: 0.5pt solid #ccc; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 0.9pt solid #555; }
img { width: 100%; margin-top: 6pt; page-break-inside: avoid; }
img[src*="fig3"] { width: 58%; display: block; margin: 6pt auto 0 auto; }
em { color: #333; }
p > em:only-child { display: block; font-size: 8.8pt; margin-bottom: 8pt; }
code { font-size: 8.8pt; }
a { color: #1c5cab; text-decoration: none; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromium", default=None, help="Chromium executable (default: Playwright's)")
    args = parser.parse_args()
    with open(os.path.join(HERE, "REPORT.md")) as f:
        body = markdown.markdown(f.read(), extensions=["tables"])
    html_path = os.path.join(HERE, "REPORT.html")
    with open(html_path, "w") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title>"
                f"<style>{CSS}</style></head><body>{body}</body></html>")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium) if args.chromium else p.chromium.launch()
        page = browser.new_page()
        page.goto("file://" + html_path)
        page.pdf(path=os.path.join(HERE, "REPORT.pdf"), format="A4", print_background=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template="<div style='font-size:8pt;width:100%;text-align:center;color:#666'>"
                                 "<span class='pageNumber'></span></div>",
                 margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
        browser.close()
    os.remove(html_path)
    print("wrote", os.path.join(HERE, "REPORT.pdf"))


if __name__ == "__main__":
    main()
