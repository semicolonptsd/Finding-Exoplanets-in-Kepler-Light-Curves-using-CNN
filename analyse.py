"""
analyse.py
----------
Step 3: figures, candidate predictions and the 'physics' numbers for the report.

    python analyse.py

Writes everything into results/ and figures/.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, precision_recall_curve

import build_dataset
from lightcurve import stitch_quarters, clean_light_curve, phase_time, make_views
from model import AstroNet

warnings.filterwarnings("ignore")
FIG = "figures"


def download_quarters(kepid):
    """Cached real light curve (or a simulated one if this is an offline test run)."""
    if str(np.load("data/views.npz")["source"]) == "simulated":
        import simulate
        return simulate.download_quarters(kepid)
    return build_dataset.download_quarters(kepid)
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b2a"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def load():
    d = np.load("data/views.npz")
    table = pd.read_csv("data/koi_used.csv")
    return d["global_view"], d["local_view"], table


def load_model():
    m = AstroNet()
    m.load_state_dict(torch.load("results/model.pt"))
    m.eval()
    return m


def predict(model, g, l):
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(g), torch.from_numpy(l))).numpy()


# ---------------------------------------------------------------------------
# 1. the KOI table (EDA)
# ---------------------------------------------------------------------------
def fig_dataset():
    koi = pd.read_csv("data/koi_table.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for disp, c in [("FALSE POSITIVE", ORANGE), ("CANDIDATE", AQUA), ("CONFIRMED", BLUE)]:
        k = koi[koi.koi_disposition == disp]
        ax[0].scatter(k.koi_period, k.koi_depth, s=3, alpha=0.35, color=c,
                      label=f"{disp.title()} ({len(k)})", linewidths=0)
    ax[0].set(xscale="log", yscale="log", xlim=(0.2, 3000), xlabel="orbital period (days)",
              ylabel="transit depth (ppm)", title="Every KOI in the catalogue")
    ax[0].legend(markerscale=4, fontsize=7, loc="upper left")

    fp = koi[koi.koi_disposition == "FALSE POSITIVE"]
    flags = {"not transit-like\n(nt)": fp.koi_fpflag_nt.sum(),
             "stellar eclipse\n(ss)": fp.koi_fpflag_ss.sum(),
             "centroid offset\n(co)": fp.koi_fpflag_co.sum(),
             "ephemeris match\n(ec)": fp.koi_fpflag_ec.sum()}
    cols = [ORANGE, ORANGE, GREY, GREY]
    ax[1].bar(list(flags), list(flags.values()), color=cols, width=0.6)
    for i, v in enumerate(flags.values()):
        ax[1].text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
    ax[1].set(title="Why false positives were rejected", ylabel="number of KOIs")
    ax[1].tick_params(axis="x", labelsize=7)
    ax[1].grid(axis="x", visible=False)
    save(fig, "dataset.png")


# ---------------------------------------------------------------------------
# 2. cleaning, step by step, on one real star
# ---------------------------------------------------------------------------
def fig_cleaning(row):
    quarters = download_quarters(row.kepid)
    raw_t = np.concatenate([q[0][q[2] == 0] for q in quarters])
    raw_f = np.concatenate([q[1][q[2] == 0] for q in quarters])
    ok = np.isfinite(raw_f) & np.isfinite(raw_t)
    st_t, st_f = stitch_quarters(quarters)
    cl_t, cl_f = clean_light_curve(quarters, row.koi_period, row.koi_time0bk, row.koi_duration)

    fig, ax = plt.subplots(3, 1, figsize=(8, 5.6), sharex=True)
    ax[0].plot(raw_t[ok], raw_f[ok] / np.nanmedian(raw_f), ".", ms=0.4, color=GREY)
    ax[0].set_title("1. Raw PDCSAP flux: each quarter sits at its own level (roll jumps)", loc="left")
    ax[1].plot(st_t, st_f, ".", ms=0.4, color=GREY)
    ax[1].set_title("2. Each quarter divided by its median: jumps gone, star's own wobble remains", loc="left")
    ax[2].plot(cl_t, (cl_f - 1) * 1e6, ".", ms=0.4, color=GREY)
    intr = np.abs(phase_time(cl_t, row.koi_period, row.koi_time0bk)) < row.koi_duration / 48
    ax[2].plot(cl_t[intr], (cl_f[intr] - 1) * 1e6, ".", ms=1.2, color=BLUE)
    ax[2].set_title("3. Detrended + upward outliers clipped: transits (blue) are what is left", loc="left")
    ax[2].set_ylabel("ppm")
    ax[2].set_xlabel("time (BKJD days)")
    for a in ax[:2]:
        a.set_ylabel("relative flux")
    save(fig, "cleaning.png")
    return cl_t, cl_f


# ---------------------------------------------------------------------------
# 3. folding, sqrt(N), and the two views
# ---------------------------------------------------------------------------
def fig_sqrt_n(row, t, f):
    """Fold the first N transits only and measure depth / scatter of the folded dip."""
    P, t0, dur = row.koi_period, row.koi_time0bk, row.koi_duration / 24
    epoch = np.round((t - t0) / P)
    x = phase_time(t, P, t0)
    first = epoch.min()
    Ns, snr = [], []
    out_noise = np.std(f[np.abs(x) > 2 * dur])
    for N in np.unique(np.geomspace(1, epoch.max() - first + 1, 25).astype(int)):
        sel = (epoch < first + N)
        inside = sel & (np.abs(x) < 0.3 * dur)
        if inside.sum() < 3:
            continue
        depth = 1 - np.mean(f[inside])
        n_tr = len(np.unique(epoch[inside]))
        Ns.append(n_tr)
        snr.append(depth / (out_noise / np.sqrt(inside.sum())))
    Ns, snr = np.array(Ns), np.array(snr)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(Ns, snr, "o", ms=4, color=BLUE, label="measured on the real star")
    c = np.median(snr / np.sqrt(Ns))
    nn = np.linspace(1, Ns.max(), 100)
    ax.plot(nn, c * np.sqrt(nn), color=INK, lw=1.2, ls="--", label=r"$\propto\sqrt{N}$")
    ax.set(xscale="log", yscale="log", xlabel="number of transits folded, N",
           ylabel="signal-to-noise of folded dip", title=f"{row.kepoi_name}: folding beats noise")
    ax.legend(fontsize=7)
    save(fig, "sqrt_n.png")


def fig_views(examples):
    """examples: list of (row, global, local, t, f, label)"""
    fig, ax = plt.subplots(len(examples), 3, figsize=(9, 2.3 * len(examples)),
                           gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    for i, (row, g, l, t, f, label, c) in enumerate(examples):
        x = phase_time(t, row.koi_period, row.koi_time0bk)
        ax[i, 0].plot(x / row.koi_period, (f - 1) * 1e6, ".", ms=0.3, color=GREY, alpha=0.5)
        lo, hi = np.percentile((f - 1) * 1e6, [0.05, 99.9])
        ax[i, 0].set_ylim(lo * 1.1, hi)
        ax[i, 0].set_ylabel("ppm")
        ax[i, 0].set_title(f"{row.kepoi_name} ({label}): all points folded", loc="left", fontsize=8)
        ax[i, 1].plot(np.linspace(-0.5, 0.5, len(g)), g, color=c, lw=0.8)
        ax[i, 1].set_title("global view (2001 bins, whole orbit)", loc="left", fontsize=8)
        ax[i, 2].plot(np.linspace(-4, 4, len(l)), l, color=c, lw=1.0)
        ax[i, 2].set_title("local view (201 bins)", loc="left", fontsize=8)
    ax[-1, 0].set_xlabel("phase")
    ax[-1, 1].set_xlabel("phase")
    ax[-1, 2].set_xlabel("time from mid-transit (durations)")
    save(fig, "views.png")


# ---------------------------------------------------------------------------
# 4. training + evaluation
# ---------------------------------------------------------------------------
def fig_training():
    h = pd.read_csv("results/history.csv")
    m = json.load(open("results/metrics.json"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    ax[0].plot(h.epoch, h.train_loss, color=BLUE, label="training loss")
    ax[0].plot(h.epoch, h.val_loss, color=ORANGE, label="validation loss")
    ax[0].axvline(m["best_epoch"], color=GREY, ls=":", lw=1)
    ax[0].text(m["best_epoch"], ax[0].get_ylim()[1] * 0.95, " weights kept", fontsize=7, va="top")
    ax[0].set(xlabel="epoch", ylabel="binary cross-entropy", title="Learning curves")
    ax[0].legend(fontsize=7)
    ax[1].plot(h.epoch, h.val_auc, color=AQUA)
    ax[1].set(xlabel="epoch", ylabel="AUC", title="Validation AUC")
    save(fig, "training.png")


def fig_evaluation():
    p = pd.read_csv("results/test_predictions.csv")
    m = json.load(open("results/metrics.json"))
    G, L, table = load()
    X_index = table.kepoi_name.isin(p.kepoi_name)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    # ROC
    fpr, tpr, _ = roc_curve(p.y, p.p_cnn)
    ax[0].plot(fpr, tpr, color=BLUE, lw=1.6, label=f"our CNN (AUC {m['cnn']['auc']:.3f})")
    if "koi_score" in m:
        s = p[np.isfinite(p.koi_score)]
        f2, t2, _ = roc_curve(s.y, s.koi_score)
        ax[0].plot(f2, t2, color=AQUA, lw=1.2,
                   label=f"NASA koi_score (AUC {m['koi_score']['auc']:.3f}, {len(s)} scored)")
    ax[0].plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=":")
    ax[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve (test set)")
    ax[0].legend(fontsize=7, loc="lower right")
    # probability histograms
    bins = np.linspace(0, 1, 21)
    ax[1].hist(p.p_cnn[p.y == 1], bins, color=BLUE, alpha=0.75, label="confirmed planets")
    ax[1].hist(p.p_cnn[p.y == 0], bins, color=ORANGE, alpha=0.75, label="false positives")
    ax[1].axvline(0.5, color=INK, lw=0.8, ls="--")
    ax[1].set(xlabel="predicted probability of planet", ylabel="count", title="What the CNN outputs")
    ax[1].legend(fontsize=7)
    # confusion matrix
    cm = np.array(m["cnn"]["confusion"])
    ax[2].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[2].text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=12)
    ax[2].set_xticks([0, 1], ["pred. FP", "pred. planet"])
    ax[2].set_yticks([0, 1], ["true FP", "true planet"])
    ax[2].grid(False)
    ax[2].set_title("Confusion matrix (threshold 0.5)")
    save(fig, "evaluation.png")

    if "koi_score" in m:
        fig, ax = plt.subplots(figsize=(4, 3.2))
        for y, c, lab in [(1, BLUE, "confirmed"), (0, ORANGE, "false positive")]:
            s = p[p.y == y]
            ax.scatter(s.koi_score, s.p_cnn, s=8, color=c, alpha=0.6, label=lab, linewidths=0)
        ax.set(xlabel="NASA koi_score", ylabel="our CNN probability", title="Us vs NASA, per KOI")
        ax.legend(fontsize=7)
        save(fig, "vs_koiscore.png")


def fig_mistakes(model):
    """The test KOIs the CNN got most confidently wrong - good for discussion."""
    p = pd.read_csv("results/test_predictions.csv")
    G, L, table = load()
    p["err"] = np.abs(p.y - p.p_cnn)
    worst = p.sort_values("err", ascending=False).head(4)
    fig, ax = plt.subplots(2, 4, figsize=(10, 3.8))
    for j, r in enumerate(worst.itertuples()):
        i = np.where(table.kepoi_name == r.kepoi_name)[0][0]
        c = BLUE if r.y == 1 else ORANGE
        ax[0, j].plot(G[i], color=c, lw=0.6)
        ax[1, j].plot(L[i], color=c, lw=0.9)
        truth = "planet" if r.y == 1 else "false pos."
        ax[0, j].set_title(f"{r.kepoi_name}: truly {truth}\nCNN says p = {r.p_cnn:.2f}", fontsize=7.5)
        for a in ax[:, j]:
            a.set_xticks([])
    ax[0, 0].set_ylabel("global")
    ax[1, 0].set_ylabel("local")
    save(fig, "mistakes.png")
    return worst


def fig_ttv(name="K00142.01"):
    """
    Measure every single transit time of one planet and compare with a strictly
    periodic ephemeris t0 + n*P ('observed minus calculated', O-C). Folding assumes O-C = 0.
    """
    koi = pd.read_csv("data/koi_table.csv").set_index("kepoi_name")
    if name not in koi.index:
        return None
    k = koi.loc[name]
    t, f = stitch_quarters(download_quarters(int(k.kepid)))
    trend = pd.Series(f, index=pd.to_timedelta(t, unit="D")).rolling(
        pd.Timedelta(days=2), center=True, min_periods=10).median().to_numpy()
    f = f / trend - 1
    P, t0, T, depth = k.koi_period, k.koi_time0bk, k.koi_duration / 24, k.koi_depth / 1e6
    shifts = np.arange(-4, 4, 0.005)
    epochs, oc = [], []
    for e in range(int((t.max() - t0) / P) + 1):
        tc = t0 + e * P
        m = np.abs(t - tc) < 4.5
        if m.sum() < 100:
            continue
        tt, ff = t[m] - tc, f[m]
        # slide a box of the transit's duration; where is the mean flux lowest?
        box = [ff[np.abs(tt - s) < T / 2].mean() if (np.abs(tt - s) < T / 2).sum() > 3 else 0
               for s in shifts]
        i = int(np.argmin(box))
        if box[i] < -0.5 * depth:
            epochs.append(e)
            oc.append(shifts[i] * 24)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    ax.plot(epochs, oc, "o", ms=3.5, color=BLUE)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.set(xlabel="transit number", ylabel="O - C (hours)",
           title=f"{name}: when each transit really happened")
    save(fig, "ttv.png")
    return float(max(oc) - min(oc))


def fig_kernels(model):
    w = model.global_branch[0][0].weight.detach().numpy()[:, 0, :]   # (16, 5)
    fig, ax = plt.subplots(2, 8, figsize=(8, 2.2), sharey=True)
    for k, a in enumerate(ax.flat):
        a.bar(range(5), w[k], color=BLUE if w[k].sum() >= 0 else ORANGE, width=0.7)
        a.set_xticks([])
        a.set_title(f"k{k+1}", fontsize=7)
        a.grid(False)
    fig.suptitle("The 16 learned first-layer kernels of the global branch (5 weights each)", fontsize=8)
    save(fig, "kernels.png")


# ---------------------------------------------------------------------------
# 5. candidates + physics
# ---------------------------------------------------------------------------
def physics(df):
    """
    From depth and catalogue stellar parameters:
      Rp = Rstar * sqrt(depth)                       (transit depth = (Rp/Rs)^2)
      a  = (M * P_yr^2)^(1/3) AU                     (Kepler's third law, solar units)
      T_eq = Teff * sqrt(Rs / 2a) (zero albedo)      (how hot the planet is)
    """
    rp = df.koi_srad * np.sqrt(df.koi_depth / 1e6) * 109.1        # Earth radii
    a = (df.koi_smass * (df.koi_period / 365.25) ** 2) ** (1 / 3)  # AU
    teq = df.koi_steff * np.sqrt(df.koi_srad / (2 * a * 215.03))
    return rp, a, teq


def candidates(model):
    G, L, table = load()
    c = table.koi_disposition == "CANDIDATE"
    out = table[c].copy()
    out["p_cnn"] = predict(model, G[c.to_numpy()], L[c.to_numpy()])
    out["rp_earth"], out["a_au"], out["teq_k"] = physics(out)
    out = out.sort_values("p_cnn", ascending=False)
    cols = ["kepoi_name", "p_cnn", "koi_score", "koi_period", "koi_depth",
            "rp_earth", "koi_prad", "a_au", "teq_k"]
    out[cols].to_csv("results/candidate_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.hist(out.p_cnn, np.linspace(0, 1, 21), color=AQUA)
    ax.set(xlabel="CNN probability of planet", ylabel="number of candidates",
           title=f"{len(out)} unconfirmed CANDIDATE KOIs")
    save(fig, "candidates.png")

    # planets the CNN is confident about, to sanity check the physics against the catalogue
    p = pd.read_csv("results/test_predictions.csv").merge(
        table[["kepoi_name", "koi_srad", "koi_smass", "koi_steff", "koi_prad"]], on="kepoi_name")
    p["rp_earth"], p["a_au"], p["teq_k"] = physics(p)
    good = p[(p.y == 1) & np.isfinite(p.rp_earth) & np.isfinite(p.koi_prad)]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.scatter(good.koi_prad, good.rp_earth, s=8, color=BLUE, alpha=0.7, linewidths=0)
    lim = [0.3, 40]
    ax.plot(lim, lim, color=GREY, lw=0.6)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="catalogue radius (Earth radii)", ylabel=r"our $R_\star\sqrt{\delta}$ (Earth radii)",
           title="Planet size from the dip depth")
    save(fig, "radius_check.png")
    ratio = (good.rp_earth / good.koi_prad)
    return out, float(np.median(ratio))


# ---------------------------------------------------------------------------
def pick(table, disp, name=None, **cond):
    t = table[table.koi_disposition == disp]
    if name is not None and (t.kepoi_name == name).any():
        return t[t.kepoi_name == name].iloc[0]
    for k, v in cond.items():
        t = t[t[k] == v] if not callable(v) else t[v(t)]
    return t.sort_values("koi_model_snr", ascending=False).iloc[len(t) // 4]


if __name__ == "__main__":
    G, L, table = load()
    model = load_model()
    summary = {}

    fig_dataset()

    # a star with only ONE known KOI, so no sibling planet's transits clutter the example
    n_on_star = pd.read_csv("data/koi_table.csv").groupby("kepid").size()
    planet = pick(table, "CONFIRMED",
                  koi_period=lambda t: (t.koi_period > 3) & (t.koi_period < 20) &
                  (t.koi_model_snr > 40) & (t.kepid.map(n_on_star) == 1))
    eb = pick(table, "FALSE POSITIVE", koi_fpflag_ss=1,
              koi_period=lambda t: (t.koi_period > 1) & (t.koi_period < 20))
    t_p, f_p = fig_cleaning(planet)
    fig_sqrt_n(planet, t_p, f_p)
    t_e, f_e = clean_light_curve(download_quarters(eb.kepid), eb.koi_period, eb.koi_time0bk, eb.koi_duration)
    ex = []
    for row, t, f, lab, c in [(planet, t_p, f_p, "confirmed planet", BLUE),
                              (eb, t_e, f_e, "eclipsing binary", ORANGE)]:
        i = np.where(table.kepoi_name == row.kepoi_name)[0][0]
        ex.append((row, G[i], L[i], t, f, lab, c))
    fig_views(ex)
    fig_training()
    fig_evaluation()
    worst = fig_mistakes(model)
    fig_kernels(model)
    ttv_range = fig_ttv("K00142.01")
    cand, ratio = candidates(model)

    summary.update(example_planet=planet.kepoi_name, example_eb=eb.kepoi_name,
                   example_planet_period=float(planet.koi_period),
                   example_planet_depth=float(planet.koi_depth),
                   example_eb_depth=float(eb.koi_depth),
                   n_candidates=len(cand),
                   cand_above_09=int((cand.p_cnn > 0.9).sum()),
                   cand_below_01=int((cand.p_cnn < 0.1).sum()),
                   radius_ratio_median=ratio, ttv_range_hours=ttv_range,
                   top_candidates=cand.head(8)[["kepoi_name", "p_cnn", "koi_score", "koi_period",
                                                "rp_earth", "a_au", "teq_k"]].round(3).to_dict("records"),
                   worst=worst[["kepoi_name", "y", "p_cnn", "koi_period", "koi_depth"]].to_dict("records"))
    json.dump(summary, open("results/analysis.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
