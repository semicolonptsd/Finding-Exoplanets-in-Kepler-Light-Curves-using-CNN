# Finding Exoplanets in Kepler Light Curves

A two-branch 1D convolutional network (a simplified AstroNet, Shallue & Vanderburg 2018) that looks at a
phase-folded NASA Kepler light curve and decides whether the dip is a real planet transit or a false
positive, such as an eclipsing binary.

By **Adarsh Kundu** and **Aarnav Piparsania**.

The full write-up, with all the maths, figures and results, is in [`report.pdf`](report.pdf). The slides
are in [`slides.pptx`](slides.pptx).

## How it works

1. **Data.** Download the Kepler Objects of Interest (KOI) table and about 1,700 long-cadence light curves:
   800 confirmed planets, 800 false positives and 200 unconfirmed candidates.
2. **Cleaning.** Stitch the quarters together, detrend the stellar variability, and clip outliers.
3. **Phase folding.** Stack every transit at the catalogue period. Noise averages down by about √N while the
   dip stays the same, so a faint signal becomes visible.
4. **Two views.** A *global* view of the whole orbit (2001 bins) and a *local* zoom on the transit (201 bins).
5. **CNN.** One convolutional branch per view, merged into fully connected layers, which output P(planet).
6. **Checks.** A logistic-regression baseline on the same views and a comparison with NASA's `koi_score`.

## Results

On a held-out test set of 238 KOIs, split by star so no star appears in both training and test:

| model | accuracy | AUC |
|---|---|---|
| CNN | 88.7% | 0.963 |
| logistic regression on the same views | 81.1% | 0.882 |
| NASA `koi_score` (reference only, uses far more information) | 97.6% | 0.992 |

## Running it

```
pip install -r requirements.txt
```

**Notebook (main version):** open `exoplanet_cnn.ipynb` and choose *Run All*. The processed dataset
(`data/views.npz`, `data/koi_used.csv`) is included in the repo, so the run takes a few minutes on a CPU.
To rebuild it from scratch, set `REBUILD_DATASET = True` (about 40 min of downloads).

**Standalone scripts:** the same pipeline, split into modules:

```
python build_dataset.py     # KOI table + light curve download, builds data/views.npz
python train.py             # trains the CNN, writes results/metrics.json
python analyse.py           # all figures + candidate predictions
```

`python build_dataset.py --simulated` runs everything offline on simulated Kepler-like stars. This is only
for testing the code.

Training is not bit-for-bit deterministic, so a fresh run lands within a percent or so of the numbers above.

## Files

| file | what it does |
|---|---|
| `exoplanet_cnn.ipynb` | the full pipeline in one notebook, with outputs |
| `lightcurve.py` | cleaning (stitch quarters, detrend, clip), phase folding, binning, global and local views |
| `build_dataset.py` | KOI table + light curve download (cached in `data/raw/`), builds `data/views.npz` |
| `model.py` | the two-branch CNN |
| `train.py` | split by star, Adam + early stopping, test metrics, logistic-regression baseline, `koi_score` comparison |
| `analyse.py` | figures, candidate predictions, planet radius, orbit and temperature estimates |
| `simulate.py` | simulated stars for offline testing |
| `data/` | KOI table and the processed folded views |

## Data

All light curves and the KOI catalogue come from the NASA Exoplanet Archive and MAST (Kepler mission).

## Reference

C. J. Shallue and A. Vanderburg, "Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain
around Kepler-80 and an Eighth Planet around Kepler-90", *AJ* 155, 94 (2018).
