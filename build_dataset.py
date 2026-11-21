"""
build_dataset.py
----------------
Step 1 of the project: get the data and turn every KOI into a (global, local) view pair.

    python build_dataset.py                 # real Kepler data (needs internet)
    python build_dataset.py --per-class 400 # smaller / faster run
    python build_dataset.py --simulated     # offline test with fake Kepler-like stars

What it does:
  1. Downloads the KOI cumulative table from the NASA Exoplanet Archive.
  2. Picks a balanced set of CONFIRMED and FALSE POSITIVE KOIs (+ some CANDIDATEs
     that we only predict on at the end).
  3. For each star, downloads every long-cadence quarter from MAST with Lightkurve,
     cleans it (lightcurve.py) and makes the folded views.
  4. Saves everything to data/views.npz plus a table data/koi_used.csv.

Downloads are cached per star in data/raw/, so if it crashes you can just re-run.
"""

import argparse
import json
import os
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")
import io
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from lightcurve import clean_light_curve, make_views

DATA = "data"
RAW = os.path.join(DATA, "raw")

KOI_COLUMNS = [
    "kepid", "kepoi_name", "koi_disposition",            # label
    "koi_pdisposition", "koi_score",                     # held out: NASA's own verdict
    "koi_period", "koi_time0bk", "koi_duration", "koi_depth",
    "koi_prad", "koi_model_snr", "koi_steff", "koi_srad", "koi_smass",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
]
TAP_URL = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
           "select+{cols}+from+cumulative&format=csv")


# ---------------------------------------------------------------------------
def download_koi_table():
    path = os.path.join(DATA, "koi_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    print("Downloading KOI cumulative table ...")
    url = TAP_URL.format(cols=",".join(KOI_COLUMNS))
    with urllib.request.urlopen(url, timeout=120) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.to_csv(path, index=False)
    return df


def choose_kois(koi, per_class, n_candidates, seed=0):
    """
    Balanced sample of planets and false positives.

    About the false positives: the KOI table flags WHY something is a false positive.
      fpflag_nt = 'not transit-like'      -> visible in the light curve shape
      fpflag_ss = 'stellar eclipse' (EB)  -> visible in the light curve shape
      fpflag_co = 'centroid offset'       -> the dip is on a NEIGHBOURING star; needs pixel data
      fpflag_ec = 'ephemeris match'       -> contamination from another star; needs a catalogue
    Our model only sees the light curve, so it physically cannot learn co/ec cases.
    We therefore train on false positives that have nt or ss set.
    """
    rng = np.random.default_rng(seed)
    koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])

    planets = koi[koi.koi_disposition == "CONFIRMED"]
    fps = koi[(koi.koi_disposition == "FALSE POSITIVE") &
              ((koi.koi_fpflag_nt == 1) | (koi.koi_fpflag_ss == 1))]
    cands = koi[koi.koi_disposition == "CANDIDATE"]

    def pick(df, n):
        return df.iloc[rng.permutation(len(df))[:n]]

    chosen = pd.concat([pick(planets, per_class), pick(fps, per_class),
                        pick(cands, n_candidates)])
    return chosen.reset_index(drop=True)


# ---------------------------------------------------------------------------
def download_quarters(kepid):
    """All long-cadence PDCSAP quarters for one star, cached as .npz."""
    path = os.path.join(RAW, f"{kepid}.npz")
    if os.path.exists(path):
        try:
            return list(np.load(path, allow_pickle=True)["quarters"])
        except Exception:
            os.remove(path)                  # half-written file from an interrupted run

    import lightkurve as lk
    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler",
                                  author="Kepler", cadence="long")
    tmp = tempfile.mkdtemp()                 # FITS files are big; we keep only the arrays
    try:
        lcs = search.download_all(quality_bitmask="default", download_dir=tmp)
        quarters = []
        for lc in lcs:
            # PDCSAP = Kepler pipeline's flux with spacecraft systematics already removed
            flux = lc["pdcsap_flux"].value
            quarters.append((np.asarray(lc.time.value, float), np.asarray(flux, float),
                             np.asarray(lc.quality.value)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    np.savez_compressed(path + ".tmp.npz", quarters=np.array(quarters, dtype=object))
    os.replace(path + ".tmp.npz", path)      # write-then-rename so a crash can't leave junk
    return quarters


def process_star(kepid, rows, download):
    """Clean one star once, then make views for every KOI on it (multi-planet stars)."""
    out = []
    try:
        quarters = download(kepid)
    except Exception as e:
        print(f"  KIC {kepid}: download failed ({e})")
        return out
    for _, r in rows.iterrows():
        try:
            t, f = clean_light_curve(quarters, r.koi_period, r.koi_time0bk, r.koi_duration)
            if t is None:
                continue
            g, l = make_views(t, f, r.koi_period, r.koi_time0bk, r.koi_duration)
        except Exception as e:                 # one weird star shouldn't kill the whole run
            print(f"  {r.kepoi_name}: skipped ({e})")
            continue
        if g is not None:
            out.append((r.kepoi_name, g, l))
    return out


def build(koi_sel, download, workers=8):
    # separate processes (not threads) so the cleaning maths runs on every CPU core
    # while other workers are waiting for MAST to send files
    groups = list(koi_sel.groupby("kepid"))
    results = []
    with ProcessPoolExecutor(workers) as pool:
        jobs = [pool.submit(process_star, kepid, rows, download) for kepid, rows in groups]
        for i, job in enumerate(as_completed(jobs), 1):
            results += job.result()
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} stars done, {len(results)} KOIs with views", flush=True)
    return results


def save(results, koi_sel, source):
    names = [r[0] for r in results]
    table = koi_sel.set_index("kepoi_name").loc[names].reset_index()
    np.savez_compressed(os.path.join(DATA, "views.npz"),
                        names=np.array(names),
                        global_view=np.stack([r[1] for r in results]),
                        local_view=np.stack([r[2] for r in results]),
                        source=source)
    table.to_csv(os.path.join(DATA, "koi_used.csv"), index=False)
    print(f"Saved {len(names)} KOIs ->", os.path.join(DATA, "views.npz"))
    print(table.koi_disposition.value_counts().to_string())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--simulated", action="store_true",
                    help="use fake Kepler-like stars instead of downloading")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)

    if args.simulated:
        import simulate
        koi = simulate.make_koi_table(args.per_class, args.candidates)
        koi.to_csv(os.path.join(DATA, "koi_table.csv"), index=False)
        with open(os.path.join(DATA, "sim_args.json"), "w") as fh:   # lets later scripts rebuild the same stars
            json.dump({"per_class": args.per_class, "n_candidates": args.candidates}, fh)
        sel = koi
        download = simulate.download_quarters            # fake stars instead of MAST
        source = "simulated"
    else:
        koi = download_koi_table()
        sel = choose_kois(koi, args.per_class, args.candidates)
        source = "kepler"
        download = download_quarters

    print(f"Processing {len(sel)} KOIs on {sel.kepid.nunique()} stars ...")
    save(build(sel, download, args.workers), sel, source)
