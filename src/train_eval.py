"""Iterative model development with a leakage-safe temporal protocol.

Split (by SWAN-SF partition, chronological, no active region shared):
  train = P1-P3 (2010-05 .. 2014-06), validation = P4 (2014-06 .. 2015-03), test = P5 (2015-03 .. 2018-08)
All learned preprocessing (imputation medians, scaling) is fitted inside a
Pipeline on the training partitions only. Class imbalance is handled with class
weights computed from the training labels only. Hyper-parameters AND the decision
threshold are chosen on validation (P4). The locked test partition (P5) is used once,
after the selection has been written to disk.
"""
import json, time, sys
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)
from common import ALL_COLS, LAST_COLS, TRAIN_P, VAL_P, TEST_P, SEED, slog

d = np.load("data/all_X.npz"); X_all = pd.DataFrame(d["X"].astype(np.float64), columns=list(d["cols"]))
meta = pd.read_csv("data/all_meta.csv.gz")
assert list(X_all.columns) == ALL_COLS
X_all = X_all.replace([np.inf, -np.inf], np.nan)
y_all = meta["y"].to_numpy()
tr, va, te = (meta.partition.isin(p).to_numpy() for p in (TRAIN_P, VAL_P, TEST_P))


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    p_o = tp + fn; n_o = fp + tn
    hss2 = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return dict(threshold=float(thr), precision=float(precision_score(y, yhat, zero_division=0)),
                recall=float(rec), f1=float(f1_score(y, yhat, zero_division=0)),
                tss=float(rec - fpr), hss2=float(hss2), far=float(fp / (tp + fp) if tp + fp else 0.0),
                accuracy=float(accuracy_score(y, yhat)), pr_auc=float(average_precision_score(y, p)),
                roc_auc=float(roc_auc_score(y, p)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                n=int(len(y)), positives=int(p_o))


def best_threshold(y, p, criterion="tss"):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 2000)))
    best = (-1, 0.5)
    for t in grid:
        yhat = p >= t
        tp = np.sum(yhat & (y == 1)); fn = np.sum(~yhat & (y == 1))
        fp = np.sum(yhat & (y == 0)); tn = np.sum(~yhat & (y == 0))
        if criterion == "tss":
            s = tp / (tp + fn) - fp / (fp + tn)
        else:  # f1
            s = 2 * tp / (2 * tp + fp + fn)
        if s > best[0]:
            best = (s, float(t))
    return best[1]


def lr_pipe(C, log=True):
    steps = [("slog", FunctionTransformer(slog))] if log else []
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
              ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED))]
    return Pipeline(steps)


def rf_pipe(leaf, depth, n=300):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, max_depth=depth,
                                                    max_features="sqrt", class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=SEED))])


CANDIDATES = [
    # Iteration 1 - baseline: point-in-time representation (values at the prediction cutoff)
    ("I1_LR_last", "Iteration 1", "Logistic Regression", LAST_COLS, lambda: lr_pipe(1.0, log=False),
     "24 last-value SHARP features, median imputation, standard scaling, C=1.0"),
    # Iteration 2 - refinement: temporal summary representation + transform + two model families
    ("I2_LR_temporal_C0.01", "Iteration 2", "Logistic Regression", ALL_COLS, lambda: lr_pipe(0.01),
     "144 temporal features, signed-log, median imputation, scaling, C=0.01"),
    ("I2_LR_temporal_C0.1", "Iteration 2", "Logistic Regression", ALL_COLS, lambda: lr_pipe(0.1),
     "144 temporal features, signed-log, median imputation, scaling, C=0.1"),
    ("I2_LR_temporal_C1", "Iteration 2", "Logistic Regression", ALL_COLS, lambda: lr_pipe(1.0),
     "144 temporal features, signed-log, median imputation, scaling, C=1.0"),
    ("I2_RF_leaf5", "Iteration 2", "Random Forest", ALL_COLS, lambda: rf_pipe(5, None),
     "144 temporal features, 300 trees, min_samples_leaf=5, max_depth=None"),
    ("I2_RF_leaf20", "Iteration 2", "Random Forest", ALL_COLS, lambda: rf_pipe(20, None),
     "144 temporal features, 300 trees, min_samples_leaf=20, max_depth=None"),
    ("I2_RF_leaf50_d16", "Iteration 2", "Random Forest", ALL_COLS, lambda: rf_pipe(50, 16),
     "144 temporal features, 300 trees, min_samples_leaf=50, max_depth=16"),
]

if __name__ == "__main__":
    results, preds_val = {}, {}
    for key, it, fam, cols, make, desc in CANDIDATES:
        t0 = time.time()
        model = make().fit(X_all.loc[tr, cols], y_all[tr])
        fit_s = time.time() - t0
        pv = model.predict_proba(X_all.loc[va, cols])[:, 1]
        thr = best_threshold(y_all[va], pv, "tss")
        results[key] = dict(iteration=it, family=fam, description=desc, n_features=len(cols),
                            fit_seconds=round(fit_s, 1), val=metrics(y_all[va], pv, thr),
                            val_at_0_5=metrics(y_all[va], pv, 0.5))
        preds_val[key] = pv
        joblib.dump(dict(model=model, cols=cols), f"models/{key}.joblib", compress=3)
        v = results[key]["val"]
        print(f"{key:24s} fit={fit_s:6.1f}s  val TSS={v['tss']:.3f} PR-AUC={v['pr_auc']:.3f} "
              f"ROC-AUC={v['roc_auc']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} thr={thr:.4f}", flush=True)

    # ---------------- model selection on VALIDATION ONLY ----------------
    sel = max(results, key=lambda k: (round(results[k]["val"]["tss"], 3), results[k]["val"]["pr_auc"]))
    pv = preds_val[sel]
    selection = dict(selected=sel, rule="max validation TSS (tie-break: validation PR-AUC)",
                     alert_threshold=results[sel]["val"]["threshold"],
                     high_threshold=best_threshold(y_all[va], pv, "f1"),
                     decided_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(selection, open("results/selection.json", "w"), indent=1)
    json.dump(results, open("results/validation_results.json", "w"), indent=1)
    pd.DataFrame({"partition": meta.partition[va], "y": y_all[va], **{k: v for k, v in preds_val.items()}}) \
        .to_csv("results/val_predictions.csv.gz", index=False)
    print("SELECTED:", selection, flush=True)

    # ---------------- single evaluation on the locked TEST partition ----------------
    test = {}
    for key, it, fam, cols, make, desc in CANDIDATES:
        m = joblib.load(f"models/{key}.joblib")["model"]
        pt = m.predict_proba(X_all.loc[te, cols])[:, 1]
        test[key] = metrics(y_all[te], pt, results[key]["val"]["threshold"])
        if key == sel:
            pd.DataFrame({"row": np.where(te)[0], "y": y_all[te], "p": pt}).to_csv(
                "results/test_predictions_selected.csv.gz", index=False)
        if key == "I1_LR_last":
            pd.DataFrame({"y": y_all[te], "p": pt}).to_csv("results/test_predictions_I1.csv.gz", index=False)
    json.dump(test, open("results/test_results.json", "w"), indent=1)
    print("TEST (selected):", json.dumps(test[sel], indent=1))
