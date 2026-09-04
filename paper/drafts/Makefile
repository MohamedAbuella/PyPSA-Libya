# paper/Makefile — reproducible figure export + paper build
# Run from repo root: `make -C paper all`

NETWORKS := $(wildcard ../networks/*.nc)

.PHONY: all figures paper clean

all: figures paper

figures: $(NETWORKS)
	python ../scripts/export_figures.py --networks-dir ../networks --out-dir figures

paper: paper.tex
	latexmk -pdf -interaction=nonstopmode paper.tex

clean:
	latexmk -C paper.tex
	rm -f *.aux *.log *.out *.synctex.gz
