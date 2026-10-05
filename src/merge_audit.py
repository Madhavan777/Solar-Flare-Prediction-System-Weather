"""Merge per-chunk feature files and write a data-audit summary (EDA)."""
import glob, json
import numpy as np, pandas as pd
Xs, metas = [], []
for p in range(1, 6):
    for f in sorted(glob.glob(f"data/features/P{p}_c*_X.npz")):
        d = np.load(f, allow_pickle=False); Xs.append(d["X"]); cols = list(d["cols"])
        m = pd.read_csv(f.replace("_X.npz", "_meta.csv")); m["partition"] = p; metas.append(m)
X = np.vstack(Xs); meta = pd.concat(metas, ignore_index=True)
assert len(X) == len(meta)
np.savez_compressed("data/all_X.npz", X=X, cols=np.array(cols))
meta.to_csv("data/all_meta.csv.gz", index=False)
meta["s"] = pd.to_datetime(meta["start"]); meta["e"] = pd.to_datetime(meta["end"])
out = {}
out["n_total"] = int(len(meta)); out["n_features"] = int(X.shape[1])
out["per_partition"] = meta.groupby("partition").agg(n=("y", "size"), pos=("y", "sum"),
    s_min=("s", "min"), e_max=("e", "max"), n_ar=("ar", "nunique")).astype(str).to_dict("index")
out["letter_by_folder"] = pd.crosstab(meta["goes_letter"], meta["folder"]).to_dict()
out["letter_by_partition"] = pd.crosstab(meta["partition"], meta["goes_letter"]).to_dict("index")
out["role_counts"] = meta["role"].fillna("").replace("", "none(FQ)").value_counts().to_dict()
out["n_rows_dist"] = meta["n_rows"].value_counts().head(5).to_dict()
out["window_hours"] = ((meta["e"] - meta["s"]).dt.total_seconds() / 3600).round(2).value_counts().head(5).to_dict()
out["nan_frac_mean"] = float(meta["nan_frac"].mean())
out["frac_windows_with_any_nan"] = float((meta["nan_frac"] > 0).mean())
out["bad_quality_frac_mean"] = float(meta["bad_quality_frac"].mean())
out["feature_nan_frac"] = float(np.isnan(X).mean())
# active regions shared between consecutive partitions (temporal-boundary check)
ars = {p: set(meta.loc[meta.partition == p, "ar"]) for p in range(1, 6)}
out["shared_ar"] = {f"P{a}-P{b}": len(ars[a] & ars[b]) for a in range(1, 6) for b in range(a + 1, 6)}
# minimum gap between windows of the same AR across partitions
json.dump(out, open("results/data_audit.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
