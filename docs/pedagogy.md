# IND5003 — Pedagogical Plan

Each chapter maps to one week: one 2–3 hour in-person session, short videos for
reference and project work, and self-study material.

**Format conventions**
- **Lecture** — core concepts, intuition, live coding, hands-on activity
- **Videos** — short (<8 min), one concept each, step-by-step procedures and code walkthroughs
- **Self-study** — derivations, extensions, additional examples students work through independently
- Hands-on activities use a dataset *different* from the lecture example so students must adapt

---

## Chapter 1 — Introduction to Python (2.5 hrs)

**Lecture**
- Why Python for data science; virtual environments (demo only)
- Lists, tuples, dicts, slicing — live coding
- NumPy arrays: creation, shape/axes, slicing, broadcasting
- *Hands-on:* Extract slices from a 3D array; compute row and column means
- Pandas Series and DataFrames; `.loc`/`.iloc`; filtering

**Videos**
- Setting up a virtual environment (Windows and macOS, ~5 min each)
- The slice operator visualised (~3 min)
- Reading a CSV and filtering with `.loc` (~5 min)

**Self-study**
- f-strings and string methods
- Regular expressions
- OOP: extend the `Circle` class to `Rectangle`

---

## Chapter 2 — Statistical Inference (3 hrs)

**Lecture**
- 5-step hypothesis test framework; why CIs are preferable
- Independent samples t-test: assumptions, rule of thumb; abalone example live
- Paired t-test: when and why; heart-rate example
- One-way ANOVA: SS decomposition intuitively; heifers example
- *Hands-on:* ANOVA on a new dataset — check assumptions, identify differing groups
- Contingency tables: chi-squared, Fisher's exact, odds ratio (brief)

**Videos**
- Reading `statsmodels` ANOVA output: what each row means (~6 min)
- Making and interpreting QQ-plots (~4 min)
- Computing and interpreting an odds ratio from a 2×2 table (~4 min)

**Self-study**
- ANOVA mathematical derivation (SS decomposition)
- Contrast estimation and Tukey HSD
- Kendall τ and ordinal association
- Nonparametric alternatives (Kruskal-Wallis, Mann-Whitney)

---

## Chapter 3 — Unsupervised Learning (2.5 hrs)

**Lecture**
- When and why unsupervised learning; curse of dimensionality
- PCA: covariance matrix intuition, scree plot, loading interpretation; wine example live
- Hierarchical clustering: dissimilarity measures, linkage, dendrogram reading
- *Hands-on:* Cluster wine PCA scores; try different linkage methods; silhouette scores

**Videos**
- Reading a scree plot and deciding how many PCs to retain (~4 min)
- Reading a dendrogram and choosing the number of clusters (~4 min)
- t-SNE: what perplexity does, how to check convergence (~5 min)

**Self-study**
- MDS vs PCA: formal differences
- Isolation Forest for outlier detection
- t-SNE on GloVe embeddings
- 3D interactive Plotly plots

---

## Chapter 4 — Natural Language Processing (3 hrs)

**Lecture**
- NLP challenges and terminology (corpus, token, vocabulary)
- Text preprocessing pipeline: tokenisation, stopwords, stemming vs. lemmatisation; live demo
- tf-idf: intuition and formula; cosine similarity
- Dense embeddings: word2vec intuition; analogy task demo with GloVe
- Hugging Face pipelines: sentiment analysis live demo on wine reviews
- *Hands-on:* Students run sentiment analysis on a dataset of their choosing and tabulate results

**Videos**
- Setting up the Gensim `CUSTOM_FILTER` pipeline step by step (~6 min)
- Understanding LDA output and the pyLDAvis visualisation (~6 min)
- Using Hugging Face `pipeline()` for the first time (~4 min)

**Self-study**
- Information retrieval with `gensim.similarities`
- Neural model interpretation (paper figures in chapter)
- Other Hugging Face pipeline tasks (zero-shot, summarisation)

---

## Chapter 5 — Linear Regression (3 hrs)

**Lecture**
- Estimation vs. prediction framing; simple linear regression model and OLS
- Interpreting coefficients and the model summary; Taiwan dataset live
- R² and its limits; adjusted R²
- Multiple regression: adding variables, partial coefficients, categorical dummies
- Residual analysis: what each plot pattern means
- *Hands-on:* Fit a model, make residual plots, identify curvature, apply log transformation

**Videos**
- Reading a `statsmodels` regression summary: every row explained (~7 min)
- What residual plots tell you and what to do about them (~6 min)
- Coding a broken-line (piecewise linear) term from scratch (~5 min)

**Self-study**
- Interaction terms: derivation of separate slopes
- Influential points and Cook's distance
- Standardised residuals and leverage
- Model selection extensions: splines, robust regression

---

## Chapter 6 — Time Series Analysis (2.5 hrs)

**Lecture**
- Time series features: trend, level, seasonality, cycles; housing sales example live
- Season plots and lag plots — building intuition before modeling
- Decomposition: additive vs. multiplicative; STL
- Stationarity and the ACF; differencing
- ARIMA: (p,d,q) in plain language; `auto_arima` demo; residual diagnostics
- *Hands-on:* Fit a model to a held-out dataset, generate forecasts, compute RMSE vs. seasonal naive

**Videos**
- How to make a season plot and what to look for (~4 min)
- Running `auto_arima` and reading its summary output (~6 min)
- Forecast error metrics: RMSE vs MAE vs MASE — when to use each (~4 min)

**Self-study**
- ETS model types and the full state-space table
- The Theta method
- Forecasting via STL decomposition
- Time series clustering

---

## Chapter 7 — Simulation (3 hrs)

**Lecture**
- Random variables: pmf vs. pdf; generating variates; the `default_rng` pattern
- LLN and CLT: why simulation works as an estimator; CIs on simulation output
- ABM concepts: agents, steps, schedulers; Mesa basics
- Boltzmann model v1 live: write Agent + Model classes from scratch; run 100 steps
- *Hands-on:* Students add Gini coefficient metric (v2) and observe convergence plot

**Videos**
- Mesa architecture: Agent, Model, scheduler, DataCollector relationship (~6 min)
- Using `mesa.batch_run()`: parameters and reading the results DataFrame (~5 min)
- *(Once DES section is added)* SimPy basics: resources and processes for a simple queue (~7 min)

**Self-study**
- Boltzmann v3: spatial grid, Moore neighbourhood
- N-gram language model
- Power analysis via simulation
- *(Once added)* Discrete Event Simulation examples with SimPy

---

## Chapter 8 — Supervised Learning (3 hrs)

**Lecture**
- The ML workflow: train/validation/test split discipline; the sklearn diagram
- Decision trees: splitting rules, Gini impurity, max_depth, visualising the tree; heart failure live
- Overfitting: gap between train and test metrics; need for cross-validation
- Random forests: why bagging + feature sampling reduces variance; GridSearchCV demo
- Validation curve: reading it to choose hyperparameters
- *Hands-on:* Tune `max_depth` for a random forest regressor on the Taiwan dataset; plot validation curve

**Videos**
- How `GridSearchCV` works internally — the train/CV split loop explained (~5 min)
- Interpreting LIME output for a single prediction (~5 min)
- Permutation importance vs. tree-based importance: the difference and when each misleads (~5 min)

**Self-study**
- PDP and ICE plots
- Random forest regressor (Taiwan example)
- SHAP values (referenced but not coded in the chapter)
- Other sklearn models: SVM, KNN, logistic regression

---

## Chapter 9 — Computer Vision (2 hrs, will grow as chapter expands)

**Lecture**
- Image representation: (H × W × 3) arrays, BGR vs. RGB
- Colour spaces: why HSV is more useful than RGB for colour selection
- Masks, contours, convex hulls: Messi ball example live
- *Hands-on:* Students apply colour masking to a different image to isolate an object
- Perspective transforms: sudoku and football examples
- Overview of CV tasks and the DNN model ecosystem

**Videos**
- Installing OpenCV and verifying the `opencv/samples/data/` path (~4 min)
- The colour picker → HSV conversion → mask workflow, step by step (~6 min)
- *(Once chapter is expanded)* Running an object detection model end to end (~8 min)

**Self-study**
- *(Once expanded)* Downloading and running DNN models from opencv_zoo
- Background subtraction
- OpenCV bootcamp (linked in chapter references)
