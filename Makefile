SHELL := /bin/bash

TEX_DIR := paper
TEX_SRC := $(TEX_DIR)/paper.tex
PDF_OUT := $(TEX_DIR)/paper.pdf

EXP_DIR := experiment
EXP_TEX := $(EXP_DIR)/reproduction_protocol.tex
EXP_PDF := $(EXP_DIR)/reproduction_protocol.pdf

LATEXMK ?= latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

LATEX ?= pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all paper experiment clean

all: paper experiment

paper: $(PDF_OUT)

experiment: $(EXP_PDF)

$(PDF_OUT): $(TEX_SRC)
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) $(LATEXMK_FLAGS) -output-directory=$(TEX_DIR) $(TEX_SRC); \
	else \
		$(LATEX) $(LATEX_FLAGS) -output-directory=$(TEX_DIR) $(TEX_SRC); \
		$(LATEX) $(LATEX_FLAGS) -output-directory=$(TEX_DIR) $(TEX_SRC); \
	fi

$(EXP_PDF): $(EXP_TEX)
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) $(LATEXMK_FLAGS) -output-directory=$(EXP_DIR) $(EXP_TEX); \
	else \
		$(LATEX) $(LATEX_FLAGS) -output-directory=$(EXP_DIR) $(EXP_TEX); \
		$(LATEX) $(LATEX_FLAGS) -output-directory=$(EXP_DIR) $(EXP_TEX); \
	fi

clean:
	@if command -v $(LATEXMK) >/dev/null 2>&1; then \
		$(LATEXMK) -C -output-directory=$(TEX_DIR) $(TEX_SRC); \
		$(LATEXMK) -C -output-directory=$(EXP_DIR) $(EXP_TEX); \
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
	rm -f $(EXP_DIR)/*.aux \
	      $(EXP_DIR)/*.bbl \
	      $(EXP_DIR)/*.bcf \
	      $(EXP_DIR)/*.blg \
	      $(EXP_DIR)/*.fdb_latexmk \
	      $(EXP_DIR)/*.fls \
	      $(EXP_DIR)/*.log \
	      $(EXP_DIR)/*.out \
	      $(EXP_DIR)/*.run.xml \
	      $(EXP_DIR)/*.synctex.gz \
	      $(EXP_DIR)/*.toc
