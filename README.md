# Higher-Order Equivalence AoS Paper

This repository is devoted to writing the paper for *The Annals of Statistics (AoS)*:

**Higher-Order Equivalence of Bayesian and Randomization Inference in Finite Populations**

## Build

From the repository root:

```bash
make
```

This builds `paper/paper.tex` into `paper/paper.pdf`.
The `Makefile` uses `latexmk` when available, and falls back to running `pdflatex` twice.

## Clean build artifacts

```bash
make clean
```

This removes temporary LaTeX build files (such as `.aux`, `.log`, `.fls`, and related artifacts).
