"""Reproducible synthetic telemetry, trained models, drift tests and promotion gates."""

import hashlib

import numpy as np
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Temperature", "Vibration", "Pressure", "Load"]
UNITS = ["°C", "mm/s", "bar", "%"]
MEANS = np.array([65, 3, 8, 60])
SCALES = np.array([10, 1, 1.5, 15])


def dataset(n, seed, scenario="normal", severity=1.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    if scenario == "sensor_shift":
        z[:, 0] += 1.8 * severity
        z[:, 1] += severity
    weights = np.array([0.8, 2.4, -0.7, 0.8])
    if scenario == "concept_shift":
        weights = weights * (1 - severity) + np.array([2.4, -1.8, 1.3, 0.3]) * severity
    y = (rng.uniform(size=n) < expit(z @ weights - 0.6)).astype(int)
    x = z * SCALES + MEANS
    if scenario == "missing_sensor":
        x[rng.random(n) < 0.65 * severity, 1] = np.nan
    return x, y


def train(x, y):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    ).fit(x, y)


def metrics(model, x, y, threshold=0.5):
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "n": len(y),
    }


def drift(reference, current):
    rows = []
    for i, name in enumerate(FEATURES):
        a = reference[:, i]
        a = a[np.isfinite(a)]
        b = current[:, i]
        b = b[np.isfinite(b)]
        test = ks_2samp(a, b) if len(b) else None
        lo, hi = (
            min(a.min(), b.min() if len(b) else a.min()),
            max(a.max(), b.max() if len(b) else a.max()),
        )
        bins = np.linspace(lo, hi, 17)
        ah, _ = np.histogram(a, bins)
        bh, _ = np.histogram(b, bins)
        missing = float(np.isnan(current[:, i]).mean())
        rows.append(
            {
                "name": name,
                "unit": UNITS[i],
                "ks": float(test.statistic) if test else None,
                "p": float(test.pvalue) if test else None,
                "missing": missing,
                "alert": test is None
                or test.pvalue < 0.05 / len(FEATURES)
                or missing > 0.05,
                "reference": (ah / len(a)).tolist(),
                "current": (bh / max(len(b), 1)).tolist(),
                "range": [float(lo), float(hi)],
            }
        )
    return rows


def fingerprint(x, y):
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()[:16]


def gate(champion, challenger):
    return (
        challenger["f1"] >= champion["f1"] + 0.02
        and challenger["brier"] <= champion["brier"] + 0.02
    )
