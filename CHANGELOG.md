# Changelog

## v0.2

### Content

- **Chapter 9 (Computer Vision)** — major expansion: added image transformations, masks, perspective transforms (Sudoku, football), Sobel/Canny edge detection, Hough circle detection, and deep learning model deployment via OpenCV DNN and opencv_zoo
- **All chapters** — exercises added throughout (chapters 1–9)
- **Chapter 4 (NLP)** — restructured; `gensim` removed and replaced with alternative implementations; backup of prior version kept as `04-nlp-backup.qmd`
- **Chapters 1–7** — content revised and updated for Python 3.14 compatibility
- Minor fixes: typo in chapter 2, stray `:::` in chapter 5, incorrect political-association reference removed

### Follow-Along Notebooks

- Added `notebooks/` directory with stripped Jupyter notebooks for all 9 chapters (`01-intro.ipynb` through `09-vision.ipynb`), a `README.md`, and a `strip_notebooks.py` utility

### `ind5003` Package

- Added `ind5003/nlp.py` module
- Updated `ind5003/inference.py`
