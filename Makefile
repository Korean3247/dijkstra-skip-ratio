# Makefile — Dijkstra Skip-Ratio Characterization
# =================================================
# Reproduces all tables, figures, and statistical results in the paper.
#
# Usage:
#   make reproduce-theory      # §III–IV: skip-ratio power-law fits
#   make reproduce-bootstrap   # §III:    bootstrap 95% CI for α and c
#   make reproduce-sensitivity # §VII:    source-vertex sensitivity
#   make reproduce-heap-py     # §VII:    Python indexed/d-ary heap comparison
#   make reproduce-heap-cpp    # §VII:    C++ indexed/d-ary heap comparison
#   make reproduce-tables      # §V–VI:   timing + Dial's experiments (Python)
#   make reproduce-all         # run all of the above in dependency order
#   make pdf                   # compile paper.tex → paper.pdf
#   make clean-results         # remove generated CSV/PNG (keep paper.pdf)
#
# Requirements: Python 3.11+, pip install -r requirements.txt
#               C++: g++ -O2 -std=c++17 (OpenMP for expA)

PYTHON   = python3
GPP      = g++
GPPFLAGS = -O2 -std=c++17
LATEX    = pdflatex
LATEXFLAGS = -interaction=nonstopmode

# ── Primary targets ───────────────────────────────────────────────────────────

.PHONY: reproduce-all reproduce-theory reproduce-bootstrap reproduce-sensitivity \
        reproduce-heap-py reproduce-heap-cpp reproduce-tables reproduce-real \
        reproduce-cpp-timing pdf clean-results clean-latex clean-bin help

## Run everything in the correct dependency order
reproduce-all: reproduce-theory reproduce-bootstrap reproduce-sensitivity \
               reproduce-heap-py reproduce-heap-cpp reproduce-tables \
               reproduce-real reproduce-cpp-timing

## §III–IV: Skip-ratio power-law fits across k, V, and weight distributions
##   Output: resultsTheory/skip_ratio_by_dist_*.csv
##           resultsTheory/alpha_fits_*.csv
##           resultsTheory/local_slope_*.csv
##           resultsTheory/*.png (plot figures)
reproduce-theory:
	@echo "=== §III–IV  theory_probe_v4.py (≈10–20 min) ==="
	$(PYTHON) theory_probe_v4.py

## §III: Bootstrap 95% CI for α and c (requires theory output CSV)
##   Output: resultsTheory/bootstrap_ci_*.csv
reproduce-bootstrap:
	@test -n "$$(ls resultsTheory/skip_ratio_by_dist_*.csv 2>/dev/null)" || \
	    { echo "ERROR: run 'make reproduce-theory' first to generate skip_ratio_by_dist_*.csv"; exit 1; }
	@echo "=== §III  bootstrap_ci.py ==="
	$(PYTHON) bootstrap_ci.py

## §VII: Source-vertex sensitivity (undirected ER, matches main experiment)
##   Output: resultsSourceSensitivity/source_sensitivity_*.csv
reproduce-sensitivity:
	@echo "=== §VII  source_sensitivity.py (≈2 min) ==="
	$(PYTHON) source_sensitivity.py

## §VII: Python indexed / d-ary heap comparison
##   Output: resultsHeapCompare/heap_compare_*.csv
reproduce-heap-py:
	@echo "=== §VII  dijkstra_heap_compare.py (≈5–10 min) ==="
	$(PYTHON) dijkstra_heap_compare.py

## §VII: C++ indexed / d-ary heap comparison
##   Output: resultsHeapCompare/heap_compare_cpp_*.csv
reproduce-heap-cpp: heap_compare
	@echo "=== §VII  dijkstra_heap_compare (C++, ≈2 min) ==="
	./heap_compare

## §V: Python timing experiments (heapq / Fibonacci / Dial's)
##   Output: results7/experiment7_*.csv, results9/experiment9_*.csv
reproduce-tables:
	@echo "=== §V  dijkstra_experiment7.py ==="
	$(PYTHON) dijkstra_experiment7.py
	@echo "=== §V  dijkstra_experiment9.py ==="
	$(PYTHON) dijkstra_experiment9.py

## §VI: Real-world SNAP experiments and plots
##   Experiment A: ca-GrQc C++ (downloads graph on first run via SNAP URL)
##   Experiment B: Dial's W-sensitivity C++
##   Experiment C: Python real-world (ego-Facebook, roadNet-CA)
##   Output: resultsA/, resultsB/, resultsC/, results10/
reproduce-real: expA expB
	@echo "=== §VI  experimentA (C++) ==="
	./expA
	@echo "=== §VI  experimentA plot ==="
	$(PYTHON) dijkstra_experimentA_plot.py
	@echo "=== §VI  experimentB (C++) ==="
	./expB
	@echo "=== §VI  experimentB plot ==="
	$(PYTHON) dijkstra_experimentB_plot.py
	@echo "=== §VI  experimentC (Python) ==="
	$(PYTHON) dijkstra_experimentC.py
	@echo "=== §VI  experiment10 (Python, SNAP validation) ==="
	$(PYTHON) dijkstra_experiment10.py

## §V: C++ timing experiments (heapq / Fibonacci / Dial's)
##   Output: results7/experiment7_cpp_*.csv, resultsA/experimentA_*.csv
reproduce-cpp-timing: exp7 exp9
	@echo "=== §V  experiment7 (C++) ==="
	./exp7
	@echo "=== §V  experiment9 (C++) ==="
	./exp9

# ── C++ build rules ──────────────────────────────────────────────────────────

heap_compare: dijkstra_heap_compare.cpp
	$(GPP) $(GPPFLAGS) -o $@ $<
	@echo "Built: heap_compare"

exp7: dijkstra_experiment7.cpp
	$(GPP) $(GPPFLAGS) -o $@ $<

exp9: dijkstra_experiment9.cpp
	$(GPP) $(GPPFLAGS) -o $@ $<

# experimentA requires OpenMP
expA: dijkstra_experimentA.cpp
	$(GPP) $(GPPFLAGS) -fopenmp -o $@ $<

expB: dijkstra_experimentB.cpp
	$(GPP) $(GPPFLAGS) -o $@ $<

# ── Paper PDF ────────────────────────────────────────────────────────────────

## Compile paper.tex → paper.pdf (two passes to resolve references)
pdf: paper.tex
	$(LATEX) $(LATEXFLAGS) paper.tex
	$(LATEX) $(LATEXFLAGS) paper.tex
	@echo "Built: paper.pdf"

# ── Helper targets ───────────────────────────────────────────────────────────

## (internal) Verify skip_ratio CSV exists for bootstrap
.PHONY: check-theory-csv
check-theory-csv:
	@test -n "$$(ls resultsTheory/skip_ratio_by_dist_*.csv 2>/dev/null)" || \
	    { echo "ERROR: run 'make reproduce-theory' first"; exit 1; }

## Remove generated CSV outputs and PNG plots (keeps paper.pdf and .tex)
clean-results:
	rm -f resultsTheory/*.csv resultsTheory/*.png
	rm -f resultsHeapCompare/*.csv
	rm -f resultsSourceSensitivity/*.csv
	@echo "Result CSVs and plots removed."

## Remove LaTeX build artifacts
clean-latex:
	rm -f paper.aux paper.log paper.out paper.fls paper.fdb_latexmk \
	      paper.synctex.gz paper.toc paper.blg paper.bbl

## Remove compiled C++ binaries
clean-bin:
	rm -f heap_compare exp7 exp9 expA expB

## Show this help
help:
	@echo ""
	@echo "Dijkstra Skip-Ratio — Reproduction Makefile"
	@echo "============================================"
	@echo ""
	@echo "  make reproduce-all         Run ALL experiments (full pipeline)"
	@echo ""
	@echo "  Individual targets:"
	@echo "  make reproduce-theory      §III–IV: power-law fits        (~10–20 min)"
	@echo "  make reproduce-bootstrap   §III:    bootstrap 95% CI      (needs theory)"
	@echo "  make reproduce-sensitivity §VII:    source-vertex CV      (~2 min)"
	@echo "  make reproduce-heap-py     §VII:    Python heap comparison (~5–10 min)"
	@echo "  make reproduce-heap-cpp    §VII:    C++ heap comparison    (~2 min)"
	@echo "  make reproduce-tables      §V:      Python timing experiments"
	@echo "  make reproduce-cpp-timing  §V:      C++ timing experiments"
	@echo "  make reproduce-real        §VI:     SNAP real-world experiments"
	@echo ""
	@echo "  make pdf                   Compile paper.tex → paper.pdf"
	@echo "  make clean-results         Remove generated CSVs/plots"
	@echo "  make clean-latex           Remove LaTeX build artifacts"
	@echo "  make clean-bin             Remove compiled C++ binaries"
	@echo ""
	@echo "  Prerequisites: pip install -r requirements.txt"
	@echo "  C++ requires:  g++ -O2 -std=c++17 (expA also needs -fopenmp)"
	@echo ""
	@echo "  Note: Core result CSVs are committed to Git."
	@echo "  Re-running experiments overwrites them with fresh results."
	@echo ""
