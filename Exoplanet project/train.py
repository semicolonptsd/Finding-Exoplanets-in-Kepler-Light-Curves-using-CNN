"""
train.py
--------
Step 2: train the CNN on CONFIRMED (y=1) vs FALSE POSITIVE (y=0), then test it.

    python train.py

Split: 70% train / 15% validation / 15% test, done BY STAR. A star with two planets
goes entirely into one split - otherwise the network could see the star's noise in
training and get an unfair hint at test time.

Outputs (results/):
    model.pt            best weights (chosen on validation loss)
    history.csv         loss / AUC per epoch
    test_predictions.csv
    metrics.json
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score)
from sklearn.model_selection import GroupShuffleSplit

from model import AstroNet

SEED = 42
EPOCHS = 60
BATCH = 64
LR = 1e-3
PATIENCE = 12          # stop if validation loss hasn't improved for this many epochs
OUT = "results"


def load_labelled():
    d = np.load("data/views.npz")
    table = pd.read_csv("data/koi_used.csv")
    lab = table.koi_disposition.isin(["CONFIRMED", "FALSE POSITIVE"]).to_numpy()
    y = (table.koi_disposition == "CONFIRMED").to_numpy().astype(np.float32)
    return (d["global_view"][lab], d["local_view"][lab], y[lab],
            table[lab].reset_index(drop=True), str(d["source"]))


def split_by_star(table):
    groups = table.kepid.to_numpy()
    idx = np.arange(len(table))
    a = GroupShuffleSplit(1, test_size=0.30, random_state=SEED)
    train, rest = next(a.split(idx, groups=groups))
    b = GroupShuffleSplit(1, test_size=0.50, random_state=SEED)
    v, t = next(b.split(rest, groups=groups[rest]))
    return train, rest[v], rest[t]


def predict(model, g, l):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(g), torch.from_numpy(l))
    return torch.sigmoid(logits).numpy()


def scores(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat),
                recall=recall_score(y, yhat), f1=f1_score(y, yhat),
                auc=roc_auc_score(y, p), avg_precision=average_precision_score(y, p),
                confusion=confusion_matrix(y, yhat).tolist())


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    G, L, y, table, source = load_labelled()
    tr, va, te = split_by_star(table)
    print(f"data source: {source}   train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"planet fraction in train: {y[tr].mean():.2f}")

    model = AstroNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()       # sigmoid + binary cross-entropy in one, numerically stable

    Gt, Lt, yt = map(torch.from_numpy, (G[tr], L[tr], y[tr]))
    best, best_state, wait, history = np.inf, None, 0, []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr))
        total = 0.0
        for i in range(0, len(tr), BATCH):
            b = perm[i:i + BATCH]
            g, l, yb = Gt[b], Lt[b], yt[b]
            # augmentation: a transit looks the same played backwards in time,
            # so flipping half the batch left-right is a free extra example
            flip = torch.rand(len(b)) < 0.5
            g = torch.where(flip[:, None], g.flip(1), g)
            l = torch.where(flip[:, None], l.flip(1), l)

            opt.zero_grad()
            loss = loss_fn(model(g, l), yb)
            loss.backward()                 # backprop: dL/dw for every weight
            opt.step()                      # Adam update
            total += loss.item() * len(b)

        p_val = predict(model, G[va], L[va])
        val_loss = nn.functional.binary_cross_entropy(
            torch.from_numpy(p_val).clamp(1e-6, 1 - 1e-6), torch.from_numpy(y[va])).item()
        val_auc = roc_auc_score(y[va], p_val)
        history.append(dict(epoch=epoch, train_loss=total / len(tr),
                            val_loss=val_loss, val_auc=val_auc))
        print(f"epoch {epoch:3d}  train loss {total/len(tr):.4f}  "
              f"val loss {val_loss:.4f}  val AUC {val_auc:.4f}")

        if val_loss < best:
            best, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, "model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(OUT, "history.csv"), index=False)

    # ---- test set: touched exactly once, at the very end
    p_test = predict(model, G[te], L[te])
    metrics = {"source": source, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "epochs_run": len(history), "best_epoch": int(np.argmin([h["val_loss"] for h in history]) + 1),
               "cnn": scores(y[te], p_test)}

    # ---- baseline: plain logistic regression on the same two views, no convolutions
    X = np.hstack([G, L])
    lr = LogisticRegression(C=0.1, max_iter=3000).fit(X[tr], y[tr])
    metrics["logreg"] = scores(y[te], lr.predict_proba(X[te])[:, 1])

    # ---- NASA's own pipeline score, on exactly the same test KOIs (real data only)
    # (a few KOIs have no koi_score, so we compare both models on the ones that do)
    ks = table.koi_score.to_numpy()[te]
    has = np.isfinite(ks)
    if has.sum() > 20:
        metrics["koi_score"] = scores(y[te][has], ks[has])
        metrics["cnn_on_scored"] = scores(y[te][has], p_test[has])
        metrics["n_scored"] = int(has.sum())

    out = table.iloc[te][["kepid", "kepoi_name", "koi_disposition", "koi_score",
                          "koi_period", "koi_depth", "koi_model_snr"]].copy()
    out["y"] = y[te]
    out["p_cnn"] = p_test
    out.to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: v for k, v in metrics.items() if k in ("cnn", "logreg", "koi_score")},
                     indent=2))


if __name__ == "__main__":
    train()
