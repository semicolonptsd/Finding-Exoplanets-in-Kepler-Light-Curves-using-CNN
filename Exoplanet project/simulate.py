"""
simulate.py
-----------
Fake-but-physical Kepler stars, used ONLY to test the pipeline offline
(python build_dataset.py --simulated). The real project runs on real Kepler data.

Each star gets the same things a real Kepler light curve has:
  * 17 quarters of ~90 days at 29.4-minute cadence, with gaps between quarters
  * a different gain per quarter (the 'roll jumps')
  * spots rotating on the star (slow quasi-periodic variability) + red noise
  * white photon noise, flagged bad cadences and cosmic-ray spikes
and then one of these signals:
  PLANET  : limb-darkened U-shaped transit, duration set by Kepler's third law
  FP (ss) : eclipsing binary - deep and/or V-shaped eclipses, a secondary eclipse,
            ellipsoidal wobble, sometimes found at half the true period (odd/even)
  FP (nt) : no real transit - variability or noise that a search mistook for one
"""

import numpy as np
import pandas as pd

CADENCE = 29.4 / 60 / 24          # days
QUARTER_DAYS = 90.0
N_QUARTERS = 17
T_START = 131.5                   # Kepler BKJD start

_TABLE = {}                       # kepid -> row, filled by make_koi_table


# ---------------------------------------------------------------------------
# physics helpers
# ---------------------------------------------------------------------------
def a_over_rstar(period_days, m_star, r_star):
    """Kepler's third law: a^3 = M P^2 (a in AU, P in years, M in solar masses)."""
    a_au = (m_star * (period_days / 365.25) ** 2) ** (1 / 3)
    return a_au * 215.03 / r_star          # 1 AU = 215 solar radii


def transit_duration(period, a_rs, k, b):
    """Total transit time T = (P/pi) * asin( sqrt((1+k)^2 - b^2) / a_rs )."""
    x = np.sqrt(max((1 + k) ** 2 - b ** 2, 1e-6)) / a_rs
    return period / np.pi * np.arcsin(min(x, 1.0))


def eclipse_shape(x, T, depth, k, b, u1=0.4, u2=0.25):
    """
    Flux drop at time x (from mid-transit) for a dark disc of radius ratio k
    crossing a limb-darkened star with impact parameter b.
    Small-planet approximation: drop = depth * I(mu)/<I> * overlap(x)
      I(mu) = 1 - u1(1-mu) - u2(1-mu)^2   (quadratic limb darkening -> U shape)
      overlap ramps linearly over ingress/egress (long ramps -> V shape when grazing)
    """
    half = T / 2
    s = np.clip(np.abs(x) / half, 0, 1)                  # 0 at centre, 1 at contact
    d = np.sqrt(b ** 2 + (1 - b ** 2) * s ** 2)          # distance from star centre
    mu = np.sqrt(np.clip(1 - np.minimum(d, 1) ** 2, 0, 1))
    limb = (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) / (1 - u1 / 3 - u2 / 6)

    ingress_frac = np.clip(2 * k / ((1 + k) * max(1 - b ** 2, 0.05) ** 0.5 + 2 * k), 0.05, 0.5)
    tau = ingress_frac * half
    overlap = np.clip((half - np.abs(x)) / tau, 0, 1)
    return depth * limb * overlap


# ---------------------------------------------------------------------------
# the KOI-like table
# ---------------------------------------------------------------------------
def _draw_star(rng):
    return dict(m=rng.uniform(0.7, 1.3), r=rng.uniform(0.7, 1.6),
                teff=rng.uniform(4500, 6500), noise=10 ** rng.uniform(-4.3, -3.4))


def _row(rng, kepid, kind):
    s = _draw_star(rng)
    P = 10 ** rng.uniform(np.log10(0.8), np.log10(150))
    t0 = T_START + rng.uniform(0, P)
    row = dict(kepid=kepid, kepoi_name=f"K{kepid:05d}.01", sim_kind=kind,
               koi_period=P, koi_time0bk=t0, koi_steff=s["teff"],
               koi_srad=s["r"], koi_smass=s["m"], sim_noise=s["noise"],
               koi_fpflag_nt=0, koi_fpflag_ss=0, koi_fpflag_co=0, koi_fpflag_ec=0)
    a_rs = a_over_rstar(P, s["m"], s["r"])

    if kind == "planet":
        k = 10 ** rng.uniform(np.log10(0.008), np.log10(0.15))      # Rp / Rs
        b = rng.uniform(0, 0.95)
        depth = k ** 2
        T = transit_duration(P, a_rs, k, b)
        row.update(sim_k=k, sim_b=b, koi_prad=k * s["r"] * 109.1)
    elif kind == "eb":
        k = rng.uniform(0.3, 1.0)                                    # companion star
        b = rng.uniform(0.0, 1.2)
        T = transit_duration(P, a_rs, k, min(b, 1 + k - 0.05))
        depth = min(k ** 2, 0.5) * rng.uniform(0.3, 1.0)
        dilution = 10 ** rng.uniform(-2.5, 0) if rng.random() < 0.5 else 1.0
        depth *= dilution                                            # blended EB
        row.update(sim_k=k, sim_b=b, sim_sec=rng.uniform(0.05, 0.9),
                   sim_half=rng.random() < 0.25, sim_ellip=rng.random() < 0.4,
                   sim_ecc_shift=rng.normal(0, 0.03) if rng.random() < 0.3 else 0.0,
                   koi_prad=np.sqrt(depth) * s["r"] * 109.1, koi_fpflag_ss=1)
    else:  # "nt": not transit-like
        T = rng.uniform(1.5, 8) / 24
        depth = s["noise"] * rng.uniform(0.3, 1.5)
        row.update(sim_mode=rng.choice(["noise", "sine", "glitch"]),
                   koi_prad=np.sqrt(depth) * s["r"] * 109.1, koi_fpflag_nt=1)

    n_in = (N_QUARTERS * QUARTER_DAYS / P) * (T / CADENCE)
    row.update(koi_duration=T * 24, koi_depth=depth * 1e6,
               koi_model_snr=depth / s["noise"] * np.sqrt(max(n_in, 1)))
    return row


def make_koi_table(per_class, n_candidates, seed=1):
    rng = np.random.default_rng(seed)
    rows, kepid = [], 10000
    labels = ([("CONFIRMED", "planet")] * per_class +
              [("FALSE POSITIVE", "eb")] * (per_class * 3 // 5) +
              [("FALSE POSITIVE", "nt")] * (per_class - per_class * 3 // 5))
    for _ in range(n_candidates):
        labels.append(("CANDIDATE", rng.choice(["planet", "eb", "nt"], p=[.7, .2, .1])))
    for disp, kind in labels:
        kepid += 1
        r = _row(rng, kepid, kind)
        r["koi_disposition"] = disp
        r["koi_pdisposition"] = np.nan      # NASA's verdicts don't exist for fake stars
        r["koi_score"] = np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    _TABLE.update({r.kepid: r for r in df.itertuples()})
    return df


# ---------------------------------------------------------------------------
# the light curve itself
# ---------------------------------------------------------------------------
def download_quarters(kepid):
    if not _TABLE:                     # fresh worker process: rebuild the same fake table
        import json
        make_koi_table(**json.load(open("data/sim_args.json")))
    r = _TABLE[kepid]
    rng = np.random.default_rng(kepid)
    t = np.arange(T_START, T_START + N_QUARTERS * QUARTER_DAYS, CADENCE)
    q_index = ((t - T_START) // QUARTER_DAYS).astype(int)
    keep = ((t - T_START) % QUARTER_DAYS) < QUARTER_DAYS - 1.2      # gaps for data downlink
    t, q_index = t[keep], q_index[keep]

    # --- star: rotating spots + red noise + white noise
    p_rot = rng.uniform(4, 40)
    amp = 10 ** rng.uniform(-4, -2.3)
    drift = np.cumsum(rng.normal(0, 0.02, len(t)))                  # spots evolve
    star = 1 + amp * (np.sin(2 * np.pi * t / p_rot + drift * 0.05)
                      + 0.5 * np.sin(4 * np.pi * t / p_rot + 1.3))
    red = np.cumsum(rng.normal(0, r.sim_noise * 0.02, len(t)))
    red -= pd.Series(red).rolling(2000, center=True, min_periods=1).mean().to_numpy()
    flux = star + red + rng.normal(0, r.sim_noise, len(t))

    # --- signal
    P, t0, T = r.koi_period, r.koi_time0bk, r.koi_duration / 24
    x = (t - t0 + 0.5 * P) % P - 0.5 * P
    depth = r.koi_depth / 1e6
    if r.sim_kind == "planet":
        flux -= eclipse_shape(x, T, depth, r.sim_k, r.sim_b)
    elif r.sim_kind == "eb":
        Ptrue = 2 * P if r.sim_half else P                             # found at half period?
        xt = (t - t0 + 0.5 * Ptrue) % Ptrue - 0.5 * Ptrue
        flux -= eclipse_shape(xt, T, depth, r.sim_k, min(r.sim_b, 1.0), u1=0.5, u2=0.2)
        xs = (t - t0 - (0.5 + r.sim_ecc_shift) * Ptrue + 0.5 * Ptrue) % Ptrue - 0.5 * Ptrue
        sec = r.sim_sec if not r.sim_half else rng.uniform(0.6, 0.95)  # odd/even mismatch
        flux -= eclipse_shape(xs, T, depth * sec, r.sim_k, min(r.sim_b, 1.0), u1=0.5, u2=0.2)
        if r.sim_ellip:                                               # tidally stretched star
            flux -= 0.1 * depth * np.cos(4 * np.pi * (t - t0) / Ptrue)
    else:
        if r.sim_mode == "sine":                                      # pulsation at that period
            flux += depth * 1.5 * np.cos(2 * np.pi * x / P)
        elif r.sim_mode == "glitch":                                  # a few random dips
            for tc in rng.uniform(t.min(), t.max(), rng.integers(1, 4)):
                flux -= depth * 4 * (np.abs(t - tc) < T / 2)

    # --- instrument: quarter gains, flagged cadences, cosmic rays
    gain = rng.uniform(0.9, 1.1, N_QUARTERS)
    flux *= gain[q_index]
    quality = np.zeros(len(t), int)
    bad = rng.random(len(t)) < 0.01
    quality[bad] = 1024
    flux[bad] *= rng.uniform(0.9, 1.1, bad.sum())
    rays = rng.random(len(t)) < 0.002
    flux[rays] += rng.uniform(5, 30, rays.sum()) * r.sim_noise

    return [(t[q_index == q], flux[q_index == q], quality[q_index == q])
            for q in range(N_QUARTERS)]
