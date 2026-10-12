#!/bin/sh
# Builds paper/arxiv.zip: everything arXiv needs to compile the paper (sources, style files,
# the figures it uses, and the generated .bbl). Run from paper/ after `latexmk -pdf main.tex`.
set -e
rm -rf /tmp/arxiv_paper && mkdir -p /tmp/arxiv_paper/figures
cp main.tex main.bbl references.bib acl.sty acl_natbib.bst /tmp/arxiv_paper/
for f in $(grep -o 'includegraphics[^{]*{[^}]*}' main.tex | sed 's/.*{\(.*\)}/\1/'); do
  cp ../figures/"$f" /tmp/arxiv_paper/figures/
done
rm -f arxiv.zip && (cd /tmp/arxiv_paper && zip -qr - .) > arxiv.zip
unzip -l arxiv.zip
