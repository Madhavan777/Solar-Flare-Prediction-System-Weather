"""Stream a SWAN-SF partition tarball and build one fixed-length feature row per
12-hour observation window (one MVTS file). Only information inside the
observation window (timestamps s..e) is used; the label comes from the file
name, which SWAN-SF derives from the largest GOES flare in the 24 h AFTER e.

Usage: python extract_features.py <partition_tar.gz> <out_prefix> [start_index] [count]
"""
import io, re, sys, tarfile, time
import numpy as np
import pandas as pd

SHARP = ["TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
         "TOTFZ", "MEANPOT", "EPSZ", "MEANSHR", "SHRGT45", "MEANGAM", "MEANGBT",
         "MEANGBZ", "MEANGBH", "MEANJZH", "TOTFY", "MEANJZD", "MEANALP", "TOTFX",
         "EPSY", "EPSX", "R_VALUE"]
STATS = ["last", "mean", "std", "min", "max", "slope"]
NAME_RE = re.compile(r"^(?P<cls>[^@_]+)(?:@(?P<fid>\d+):(?P<role>Primary|Secondary))?_ar(?P<ar>\d+)_s(?P<s>[^_]+)_e(?P<e>[^_]+)\.csv$")

def features(df):
    v = df[SHARP].to_numpy(dtype=np.float64)          # (T, 24) inside the window only
    t = np.arange(v.shape[0], dtype=np.float64) * 0.2  # hours (12-min cadence)
    out = np.full((len(SHARP), len(STATS)), np.nan)
    with np.errstate(all="ignore"):
        for j in range(v.shape[1]):
            col = v[:, j]
            ok = np.isfinite(col)
            if ok.sum() == 0:
                continue
            c = col[ok]; tt = t[ok]
            out[j, 0] = c[-1]
            out[j, 1] = c.mean(); out[j, 2] = c.std()
            out[j, 3] = c.min();  out[j, 4] = c.max()
            if ok.sum() >= 2 and np.ptp(tt) > 0:
                out[j, 5] = np.polyfit(tt, c, 1)[0]   # trend per hour
    nan_frac = float(np.mean(~np.isfinite(v)))
    return out.ravel(), nan_frac

def main(tar_path, out_prefix, start=0, count=10**9):
    rows, meta = [], []
    t0 = time.time()
    idx = -1
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".csv"):
                continue
            idx += 1
            if idx < start:
                continue
            if idx >= start + count:
                break
            parts = m.name.split("/")
            fname, folder = parts[-1], parts[-2]
            g = NAME_RE.match(fname)
            if g is None:
                print("skip name", fname, file=sys.stderr); continue
            raw = tf.extractfile(m).read()
            df = pd.read_csv(io.BytesIO(raw), sep="\t", usecols=["Timestamp", "QUALITY"] + SHARP)
            f, nan_frac = features(df)
            q = pd.to_numeric(df["QUALITY"], errors="coerce").fillna(0).to_numpy()
            cls = g["cls"]
            meta.append(dict(file=fname, folder=folder, flare_class=cls,
                             goes_letter=cls[0], y=int(folder == "FL"), role=g["role"] or "",
                             ar=int(g["ar"]), start=g["s"], end=g["e"],
                             n_rows=len(df), nan_frac=nan_frac,
                             bad_quality_frac=float(np.mean(q != 0)),
                             first_ts=str(df["Timestamp"].iloc[0]),
                             last_ts=str(df["Timestamp"].iloc[-1])))
            rows.append(f.astype(np.float32))
    if not rows:
        print("empty chunk", flush=True); open(out_prefix + "_EMPTY", "w").close(); return
    X = np.vstack(rows)
    cols = [f"{p}__{s}" for p in SHARP for s in STATS]
    np.savez_compressed(out_prefix + "_X.npz", X=X, cols=np.array(cols))
    pd.DataFrame(meta).to_csv(out_prefix + "_meta.csv", index=False)
    print("done", X.shape, f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 0, int(a[4]) if len(a) > 4 else 10**9)
