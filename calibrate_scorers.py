"""Fit the learned (non-fuzzy) trust functions used in the trust-function study.

The calibration federation is disjoint from every evaluation run: seeds 100 and
101 (never used elsewhere), Fashion-MNIST at alpha = 0.5, and only two attacks
(label flipping and sign flipping). Per-round, per-client statistic vectors
u = (alignment, peer agreement, magnitude regularity, temporal consistency) are
logged from the full FedEFT pipeline together with the ground-truth label
(1 = honest, 0 = Byzantine), and three scorers are fitted:

  linopt   : least-squares linear score, clipped to [0,1]
  logistic : logistic regression (maximum likelihood, L2, C = 1)
  mlp      : 4-16-16-1 ReLU network, cross-entropy, Adam

The fitted parameters are written to calib/scorers.json. The evaluation then
uses attacks the scorers never saw (scaling, ALIE, IPM, adaptive, backdoor),
which is where a hand-designed rule base and a fitted scorer can differ.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_experiments import FM  # noqa: E402

OUT = os.path.join(HERE, "calib")


def _run(args):
    attack, seed = args
    from fedhift.fl import FLConfig, run
    r = run(FLConfig(**{**FM, "aggregator": "fedhift", "attack": attack, "seed": seed,
                        "alpha": 0.5, "byz_frac": 0.2, "keep_log": True}))
    X, y = [], []
    for rec in r["weight_log"]:
        X.extend(rec["u"])
        y.extend(rec["labels"])
    return X, y


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(a, s) for a in ("label_flip", "sign_flip") for s in (100, 101)]
    with Pool(2) as p:
        res = p.map(_run, jobs)
    X = np.array([x for r in res for x in r[0]], dtype=np.float64)
    y = np.array([v for r in res for v in r[1]], dtype=np.float64)
    np.savez(os.path.join(OUT, "calib_data.npz"), X=X, y=y)

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score

    A = np.c_[X, np.ones(len(X))]
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu", solver="adam",
                        max_iter=5000, random_state=0).fit(X, y)
    params = {
        "linopt": {"coef": beta[:4].tolist(), "intercept": float(beta[4])},
        "logistic": {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
        "mlp": {"coefs": [c.tolist() for c in mlp.coefs_],
                "intercepts": [b.tolist() for b in mlp.intercepts_]},
        "n_samples": int(len(y)), "n_byzantine": int((y == 0).sum()),
        "train_auc": {
            "linopt": float(roc_auc_score(y, A @ beta)),
            "logistic": float(roc_auc_score(y, lr.decision_function(X))),
            "mlp": float(roc_auc_score(y, mlp.predict_proba(X)[:, 1])),
        },
    }
    # sanity: the numpy re-implementation must reproduce sklearn's MLP output
    from fedhift.aggregators import learned_trust
    assert np.allclose(learned_trust("mlp", X[:50], params), mlp.predict_proba(X[:50])[:, 1])
    assert np.allclose(learned_trust("logistic", X[:50], params), lr.predict_proba(X[:50])[:, 1])
    with open(os.path.join(OUT, "scorers.json"), "w") as f:
        json.dump(params, f, indent=1)
    print(json.dumps(params["train_auc"]), params["n_samples"], params["n_byzantine"])


if __name__ == "__main__":
    main()
