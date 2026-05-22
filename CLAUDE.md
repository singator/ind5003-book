# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Quarto book titled *Data Analytics for Sense-Making* (IND5003), a graduate-level data analytics course at NUS. Chapters are written as `.qmd` files using Python (Jupyter kernel). The rendered output lives at <https://singator.github.io/ind5003-book>.

## Environment setup

1. Create a Python virtual environment at `env/` in the project root.
2. Install dependencies: `pip install -r requirements.txt`
3. Install [Quarto](https://quarto.org/docs/get-started/).

The Quarto project picks up the `env/` virtual environment automatically (no reticulate needed).

## Build commands

```bash
# Render the full book to HTML and PDF
quarto render

# Render only HTML
quarto render --to html

# Render only PDF
quarto render --to pdf

# Render a single chapter (faster iteration)
quarto render 03-unsupervised.qmd

# Preview with live reload
quarto preview
```

Slides (in `slides/`) are Beamer PDFs rendered separately:
```bash
quarto render slides/01-intro-slides.qmd
```

## Project structure

| Path | Purpose |
|---|---|
| `_quarto.yml` | Book configuration: chapter order, output formats, Google Analytics |
| `index.qmd` | Preface (unnumbered chapter) |
| `01-intro.qmd` … `09-vision.qmd` | Book chapters in order |
| `references.qmd` / `references.bib` | Bibliography |
| `figs/` | Shared figures referenced across chapters |
| `slides/` | Beamer slide decks for each chapter |
| `_book/` | Build output (gitignored via Quarto conventions; present locally) |
| `.quarto/` | Quarto cache (gitignored) |
| `env/` | Python virtual environment (gitignored) |

Chapter execution uses the `jupyter: python3` kernel declared in `_quarto.yml`. Individual chapters can override this in their YAML front matter.

## Authoring conventions

- Each chapter `.qmd` starts with a YAML front matter block (`---`) containing at minimum `title:`.
- Code cells are standard Quarto/Jupyter Python cells.
- Figures are stored in `figs/` and referenced with relative paths like `figs/filename.png`.
- The PDF output uses `scrreprt` document class with a generated index (`\makeindex` / `\printindex`).
- Jupyter notebooks (`.ipynb`) in the root are working drafts — the canonical source is always the `.qmd` file.

## Key dependencies (notable packages)

- `scikit-learn`, `scipy`, `numpy`, `pandas` — core ML/stats
- `gensim`, `transformers`, `sentence-transformers` — NLP
- `pyLDAvis` — topic model visualization
- `Mesa` — agent-based simulation
- `pmdarima` — time series
- `opencv-python`, `scikit-image` — computer vision
- `folium`, `geopandas` — geospatial
