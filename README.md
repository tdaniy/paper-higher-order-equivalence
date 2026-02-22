# Higher-Order Equivalence AoS Paper

This repository is devoted to writing the paper for *The Annals of Statistics (AoS)*:

**Higher-Order Equivalence of Bayesian and Randomization Inference in Finite Populations**

## Build

From the repository root:

```bash
make
```

This builds:
- `paper/paper.tex` into `paper/paper.pdf`
- `experiment/reproduction_protocol.tex` into `experiment/reproduction_protocol.pdf`

You can build targets individually:

```bash
make paper
make experiment
```

The `Makefile` uses `latexmk` when available, and falls back to running `pdflatex` twice.

## Experiments

The `experiment/` folder contains the reproducibility protocol and the self-contained reproducible environment for numerical experiments. See `experiment/README.md` for details.

## Citation

See `CITATION.cff` for citation metadata.

## Clean build artifacts

```bash
make clean
```

This removes temporary LaTeX build files (such as `.aux`, `.log`, `.fls`, and related artifacts).
