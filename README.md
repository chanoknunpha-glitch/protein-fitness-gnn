# Protein Fitness Prediction using Sequence–Structure Embeddings and Graph Neural Networks

Code accompanying the manuscript *"Representation Learning for Low-Data Protein
Fitness Prediction Using Sequence–Structure Embeddings and Graph Neural
Networks"* (under review at *Scientific Reports*).

Each variant is turned into a residue-level graph from a predicted structure and
a pretrained MIF-ST representation, then passed through a three-layer `NNConv`
GNN with global mean pooling and a linear regression head.

## Contents

| File / folder | What it is |
|---|---|
| `extract_mifst_clean.ipynb` | step 1 — extracts the per-residue MIF-ST representations |
| `model5_mifst.ipynb` | step 2 — builds the graphs, tunes, trains and evaluates the model |
| `pdb_utils.py`, `pretrained.py`, `collaters.py`, ... | MIF-ST source files (see `THIRD_PARTY_NOTICES.md`) |
| `SOFTWARE_ENVIRONMENT.md` | the packages used, and the baseline protocol described in the paper |
| `data/` | variants, splits, structures and extracted representations |

## Setup

```bash
conda create -n fitness-gnn python=3.11
conda activate fitness-gnn
pip install -r requirements.txt
```

`torch-geometric` must match your PyTorch build — see the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
The published results were produced on CPU; a GPU is considerably faster.

## How to run

**1. Extract the MIF-ST representations.** Open `extract_mifst_clean.ipynb`, set
the three paths in the Config cell, run all:

```
CSV_PATH = "data/demo_50_variants.csv"
PDB_DIR  = "data/PDB_AlphaFold/"
OUT_DIR  = "data/output_extract_mifst_AlphaFold/"
```

One `<name>_mifst_per_tok.pt` of shape `[286, 256]` is written per variant. The
representation depends on the sequence and the structure only, so this is done
once and reused by every split.

**2. Train and evaluate.** Open `model5_mifst.ipynb`, check the paths in the
Config cell, run all. It builds the graphs, runs the Optuna search, trains the
final model for 200 epochs and reports the metrics. A smoke-test cell runs three
epochs first, so you can check the timing before committing to the full run.

## Data

Each csv has the columns:

| column | meaning |
|---|---|
| `name` | variant identifier, e.g. `L223I` |
| `sequence` | full 286-residue amino-acid sequence |
| `fitness` | the regression target |
| `pdb` | file name inside `data/PDB_AlphaFold/` |
| `embed` | file name inside `data/output_extract_mifst_AlphaFold/` |

`train.csv`, `val.csv` and `test.csv` are disjoint: the validation set is used
for the Optuna objective and must not overlap the test set.

The fitness measurements and predicted structures used in the paper are not ours
to redistribute. What ships here instead is a public demo set:
`demo_50_variants.csv` holds 50 single mutants of TEM-1 β-lactamase drawn with
seed 42 from `BLAT_ECOLX_Stiffler_2015` in [ProteinGym](https://proteingym.org),
split 30 / 10 / 10, with one predicted structure per variant in
`data/PDB_AlphaFold/`. Please cite Stiffler et al. (2015) and the ProteinGym
benchmark if you use it.

**What the demo can and cannot show.** The demo differs from the data used in the
paper in two ways that matter.

First, it has 30 training variants rather than 122, against a model whose first
layer alone has 4.3 M parameters.

Second, and more importantly, every demo variant is a *single* mutant, so any two
sequences differ at one position out of 286. The variants in the paper carry one
to five mutations on a common background. After global mean pooling over the 286
residues, a one-residue difference is a fraction of a percent of the graph
representation, which makes the regression substantially harder than the task the
architecture was designed for.

In our own run of the demo the model does learn a usable signal — a validation
mean squared error of about 1.06, against 1.43 for simply predicting the training
mean — but the test split has only 10 variants, and a correlation estimated from
10 points has a 95% confidence interval spanning roughly −0.8 to +0.5.

The numbers this notebook produces are therefore **not** indicative of the
method's performance and are not the results reported in the paper. The demo
exists so that a reader can verify the implementation runs end to end on public
data.

## Method summary

* **Nodes** — 286 residues, features `[MIF-ST(256) | Cα xyz(3)]` = 259-d.
* **Edges** — the k = 5 nearest residues by **virtual Cβ–Cβ** distance (self
  excluded), directed (neighbour → residue), not symmetrised, no sequential
  backbone edges: 286 × 5 = 1430 edges per graph.
* **Edge features** — `[distance, ω, θ, φ | MIF-ST of the source residue(256)]`
  = 260-d. Virtual Cβ atoms are built from N, Cα and C following the trRosetta
  convention, as implemented in `pdb_utils.process_coords`.
* **Model** — `NNConv(259→128) → NNConv(128→64) → NNConv(64→32)`, each with an
  edge network `Linear(260, h) → ReLU → Linear(h, d_in·d_out)` and `aggr='mean'`,
  then global mean pooling and `Linear(32, 1)`.
* **Training** — MSE loss, Adam, `batch_size = 1`, 200 epochs. Hyperparameters
  chosen by 30 Optuna trials of 15 epochs over learning rate (1e-4–1e-3, log),
  weight decay (1e-6–1e-3, log) and dropout (0.1–0.5). The setting reported in
  the paper for the 80:20 split is lr 7.111e-4, weight decay 1.315e-6, dropout
  0.2169; the notebook has a cell that sets these directly, so the search can be
  skipped.

## Notes for anyone reproducing this

* The published runs were not seeded, so re-running gives slightly different
  numbers. Results also differ between CPU and GPU: reduction order differs and
  PyTorch Geometric's scatter operations are not deterministic on CUDA.
* The reported model is the one at the **final (200th) epoch**. The loop keeps
  the lowest-validation checkpoint as well, but does not restore it, and no early
  stopping is applied.
* The notebook trains the full model. The ablations reported in the paper are
  small changes to the same code: removing MIF-ST from the node features
  (`x = ca_coords_tensor`), removing the coordinates instead (`x = embedded`),
  dropping the three angle blocks from `edge_attr`, using the trRosetta
  structures, or using a fixed hyperparameter setting instead of the Optuna
  search.

## Licence

MIT — see `LICENSE`. The MIF-ST source files in this folder are covered by their
own licence; see `THIRD_PARTY_NOTICES.md`.

## Citation

See `CITATION.cff`. Please also cite MIF-ST (Yang et al.), AlphaFold2 (Jumper et
al., 2021) and, if you use the demo data, ProteinGym and Stiffler et al. (2015).
