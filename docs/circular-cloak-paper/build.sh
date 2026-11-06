#!/bin/sh
# Build the paper. Re-create tables first with: python make_tables.py
set -u
cd "$(dirname "$0")"
latexmk -C paper >/dev/null 2>&1 || true
latexmk -pdf -bibtex -interaction=nonstopmode -f paper.tex || true
grep -E "Citation.*undefined|Reference.*undefined|Overfull .hbox" paper.log | sort | uniq -c | head -20
latexmk -c paper >/dev/null 2>&1 || true   # keep the PDF and bbl, drop the rest
