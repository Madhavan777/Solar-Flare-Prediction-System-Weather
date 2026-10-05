import json, joblib, numpy as np, pandas as pd
from common import ALL_COLS

sel = json.load(open("results/selection.json"))
key, thr, hi = sel["selected"], sel["alert_threshold"], sel["high_threshold"]
bundle = joblib.load(f"models/{key}.joblib")
model, cols = bundle["model"], bundle["cols"]
meta = pd.read_csv("data/all_meta.csv.gz")
d = np.load("data/all_X.npz")
X = pd.DataFrame(d["X"].astype(np.float64), columns=list(d["cols"]))
te = meta.partition == 5
Xte, mte = X.loc[te, cols], meta.loc[te].reset_index(drop=True)
p = model.predict_proba(Xte)[:, 1]
mte["p"] = p

tp = mte[(mte.y == 1) & (mte.p >= hi)].sort_values("p", ascending=False).iloc[0]
tn = mte[(mte.y == 0) & (mte.p < thr)].sample(1, random_state=3).iloc[0]

test_res = json.load(open("results/test_results.json"))[key]
val_res = json.load(open("results/validation_results.json"))
coef = model.named_steps["clf"].coef_.ravel()
order = np.argsort(np.abs(coef))[::-1][:6]

demo = dict(
    model_key=key,
    model_name="Logistic Regression (temporal features, C=0.01)",
    alert_threshold=round(thr, 4), high_threshold=round(hi, 4),
    test=test_res,
    examples=dict(
        flare_case=dict(ar=int(tp.ar), start=tp.start, end=tp.end, flare_class=tp.flare_class,
                        probability=round(float(tp.p), 4)),
        quiet_case=dict(ar=int(tn.ar), start=tn.start, end=tn.end, probability=round(float(tn.p), 4)),
    ),
    top_features=[{"name": cols[i], "weight": round(float(coef[i]), 4)} for i in order],
)
json.dump(demo, open("results/dashboard_demo.json", "w"), indent=1)
print(json.dumps(demo, indent=1))
