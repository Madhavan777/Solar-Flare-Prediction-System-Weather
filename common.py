"""Shared definitions for the Solar Flare Prediction & Space Weather Alert System.

Forecast formulation (fixed by SWAN-SF and verified on the downloaded files):
  * observation window : 12 h of SHARP data (60 records, 12-min cadence)
  * prediction cutoff  : the last timestamp of the observation window
  * prediction window  : the following 24 h
  * target y = 1       : the window file sits in the SWAN-SF 'FL' folder, i.e. the
                         largest GOES flare of the active region in the next 24 h is M or X
    target y = 0       : 'NF' folder (flare-quiet, B- or C-class maximum)
Only the 24 SHARP magnetic-field parameters observed inside the window are used.
"""
import numpy as np

SHARP = ["TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
         "TOTFZ", "MEANPOT", "EPSZ", "MEANSHR", "SHRGT45", "MEANGAM", "MEANGBT",
         "MEANGBZ", "MEANGBH", "MEANJZH", "TOTFY", "MEANJZD", "MEANALP", "TOTFX",
         "EPSY", "EPSX", "R_VALUE"]
STATS = ["last", "mean", "std", "min", "max", "slope"]
ALL_COLS = [f"{p}__{s}" for p in SHARP for s in STATS]
LAST_COLS = [f"{p}__last" for p in SHARP]

# Columns present in the SWAN-SF files that are deliberately NOT used as predictors.
EXCLUDED = {
    "label/flare-derived (leakage risk)": ["BFLARE", "CFLARE", "MFLARE", "XFLARE",
        "BFLARE_LOC", "CFLARE_LOC", "MFLARE_LOC", "XFLARE_LOC",
        "BFLARE_LABEL", "CFLARE_LABEL", "MFLARE_LABEL", "XFLARE_LABEL",
        "BFLARE_LABEL_LOC", "CFLARE_LABEL_LOC", "MFLARE_LABEL_LOC", "XFLARE_LABEL_LOC"],
    "GOES X-ray measurements": ["XR_MAX", "XR_QUAL"],
    "position / geometry metadata": ["CRVAL1", "CRLN_OBS", "CRLT_OBS", "CRVAL2", "HC_ANGLE",
        "LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"],
    "quality / flags (used for auditing only)": ["QUALITY", "SPEI", "IS_TMFI"],
}

TRAIN_P, VAL_P, TEST_P = (1, 2, 3), (4,), (5,)
SEED = 42


def slog(x):
    """Signed log transform: fixed (not learned), so it cannot leak information."""
    return np.sign(x) * np.log10(1.0 + np.abs(x))
