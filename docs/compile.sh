#!/bin/bash
# Compile simulation_documentation.tex to PDF
# Run twice for correct cross-references and TOC

cd "$(dirname "$0")"

echo "[1/2] First pass..."
pdflatex -interaction=nonstopmode simulation_documentation.tex

echo "[2/2] Second pass (cross-references)..."
pdflatex -interaction=nonstopmode simulation_documentation.tex

echo ""
echo "Done! Output: $(ls -lh simulation_documentation.pdf)"
