SHELL := /bin/bash

TEX_DIR := paper
TEX_SRC := $(TEX_DIR)/paper.tex
PDF_OUT := $(TEX_DIR)/paper.pdf

LATEXMK ?= latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(TEX_DIR)

LATEX ?= pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(TEX_DIR)

.PHONY: all clean

all: $(PDF_OUT)

$(PDF_OUT): $(TEX_SRC)
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) $(LATEXMK_FLAGS) $(TEX_SRC); \
	else \
		$(LATEX) $(LATEX_FLAGS) $(TEX_SRC); \
		$(LATEX) $(LATEX_FLAGS) $(TEX_SRC); \
	fi

clean:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -C -output-directory=$(TEX_DIR) $(TEX_SRC); \
	fi
	rm -f $(TEX_DIR)/*.aux \
	      $(TEX_DIR)/*.bbl \
	      $(TEX_DIR)/*.bcf \
	      $(TEX_DIR)/*.blg \
	      $(TEX_DIR)/*.fdb_latexmk \
	      $(TEX_DIR)/*.fls \
	      $(TEX_DIR)/*.log \
	      $(TEX_DIR)/*.out \
	      $(TEX_DIR)/*.run.xml \
	      $(TEX_DIR)/*.synctex.gz \
	      $(TEX_DIR)/*.toc
