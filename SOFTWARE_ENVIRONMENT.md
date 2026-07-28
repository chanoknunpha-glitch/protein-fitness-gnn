# Software environment

Text for the manuscript appendix. Package versions are pinned in
`requirements.txt`.

## B. Software environment

All experiments were implemented in Python 3.11. The following software packages
were used.

**Deep learning framework.** All neural models are implemented in `PyTorch` [n].
Graph construction, message passing and batching use `PyTorch Geometric` [n]; the
graph convolution is its `NNConv` operator, which maps each edge feature vector to
a weight matrix through a learned edge network, followed by `global_mean_pool` for
graph-level readout.

**Protein representations.** Per-residue representations are obtained from the
pretrained MIF-ST model [n] using the reference implementation in the
`sequence-models` package [n], which also supplies the structure parser and the
geometric features — virtual Cβ–Cβ distances and the ω, θ and φ inter-residue
angles — through its `pdb_utils` module. ESM-2 (650M) representations, used for
comparison, are extracted with the `fair-esm` package [n] with all weights frozen.

**Protein structures.** Structures are predicted models: AlphaFold2 [n] for the
main experiments and trRosetta [n] for the structure-source comparison. Only the
backbone N, Cα and C coordinates are read.

**Hyperparameter optimization.** Hyperparameters are selected with `Optuna` [n],
using its Tree-structured Parzen Estimator sampler with a fixed seed, over 30
trials of 15 epochs per configuration.

**Baseline regressors.** Gradient-boosted trees are implemented with `XGBoost`
[n] using the squared-error objective. The multilayer perceptron baseline is
written directly in `PyTorch` and uses `scikit-learn` [n] for input
standardization.

**Evaluation and data handling.** Coefficient of determination, mean squared
error and mean absolute error are computed with `scikit-learn` [n]; Pearson and
Spearman correlations with `SciPy` [n]. Array and tabular operations use `NumPy`
[n] and `pandas` [n], and figures are produced with `Matplotlib` [n] and `seaborn`
[n].

## C. Baseline protocol

**Feature construction.** Each baseline consumes the same per-residue
representations as the graph model, reduced to a single fixed-length vector per
variant by averaging over the residue axis. This yields a 256-dimensional vector
for MIF-ST and a 1280-dimensional vector for ESM-2. Because pooling discards the
spatial arrangement of residues, the baselines retain the sequence and structure
information captured by the pretrained model but lose the explicit geometry that
the graph model uses.

**Gradient-boosted trees.** An `XGBRegressor` is fitted to the pooled features.
In the tuned variant, `Optuna` searches the number of estimators (100–800),
maximum depth (2–8), learning rate (10⁻³–3×10⁻¹, log scale), subsample and
column-subsample fractions (0.5–1.0), and the L1 and L2 regularization strengths
(10⁻⁴–10 and 10⁻³–10, log scale) for 30 trials, after which the best
configuration is refitted on the full training set.

**Multilayer perceptron.** The MLP consists of one to three hidden layers, each a
linear transformation followed by a ReLU activation and dropout, with a single
linear output unit. Inputs are standardized to zero mean and unit variance using
statistics computed on the training set only. Training uses the Adam optimizer
with a mean squared error loss for 200 epochs in full-batch mode. The tuned
variant searches the hidden width (64, 128 or 256), the number of layers (1–3),
dropout (0.1–0.5), learning rate (10⁻⁴–10⁻², log scale) and weight decay
(10⁻⁶–10⁻³, log scale) for 30 trials.

**Selection protocol.** For both baselines the search objective is the mean
squared error on a held-out 15% of the training set, drawn with a fixed seed. The
test split is used only for the final evaluation and never influences model
selection.

**Untuned variants.** To separate the contribution of hyperparameter search from
that of the representation, each baseline is additionally run with fixed
settings, reported in the supplementary material. The trees use 300 estimators,
maximum depth 4, learning rate 0.05 and subsample and column-subsample fractions
of 0.8. The perceptron uses two hidden layers of width 128, dropout 0.2, learning
rate 10⁻⁴ and weight decay 10⁻⁴, mirroring the untuned graph-model configuration.

---

## References to fill in

| Placeholder | Reference |
|---|---|
| PyTorch | Paszke et al., NeurIPS 2019 |
| PyTorch Geometric | Fey & Lenssen, ICLR-W 2019 |
| MIF-ST | Yang et al. |
| sequence-models | github.com/microsoft/protein-sequence-models |
| ESM-2 / fair-esm | Lin et al., Science 2023 |
| AlphaFold2 | Jumper et al., Nature 2021 |
| trRosetta | Yang et al., PNAS 2020 |
| Optuna | Akiba et al., KDD 2019 |
| XGBoost | Chen & Guestrin, KDD 2016 |
| scikit-learn | Pedregosa et al., JMLR 2011 |
| SciPy | Virtanen et al., Nat. Methods 2020 |
| NumPy | Harris et al., Nature 2020 |
| pandas | McKinney, SciPy 2010 |
| Matplotlib | Hunter, CiSE 2007 |
| seaborn | Waskom, JOSS 2021 |
