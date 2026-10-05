# Data card — SWAN-SF as used in this project

## 1. Provenance and citation

**Dataset:** SWAN-SF (Space Weather ANalytics for Solar Flares), version 1.2.
**DOI:** [10.7910/DVN/EBCFKM](https://doi.org/10.7910/DVN/EBCFKM), Harvard Dataverse.
**Reference paper:** Angryk, R. A., Martens, P. C., Aydin, B., Kempton, D., Mahajan, S. S.,
Basodi, S., Ahmadzadeh, A., Cai, X., Filali Boubrahimi, S., Hamdi, S. M., Schuh, M. A., and
Georgoulis, M. K. (2020). "Multivariate time series dataset for space weather data analytics."
*Scientific Data* **7**, 227. [doi:10.1038/s41597-020-0548-x](https://doi.org/10.1038/s41597-020-0548-x)

The dataset is **not redistributed** by this repository. It must be downloaded from the Dataverse
and is covered by that repository's terms, not by this project's MIT licence. The trained model
artefacts in `models/` are derived parameters, not copies of the data.

Underlying observations are SDO/HMI Space-weather HMI Active Region Patch (SHARP) products, produced
by the HMI team at Stanford's Joint Science Operations Center (JSOC).

## 2. What one row is

One **window** = one SWAN-SF multivariate time-series file = one observation of one active region:

| Property | Value |
|---|---|
| Records per window | 60, every one of the 331,185 windows |
| Cadence | 12 minutes |
| First-to-last timestamp span | 11.8 h (59 × 0.2 h) |
| Nominal observation window | 12 h |
| Prediction cutoff | the window's last timestamp |
| Prediction window | the following 24 h |

The nominal 12 h and the measured 11.8 h span are both correct: 60 samples at a 12-minute cadence
cover a 12-hour interval but span 11.8 h from first to last sample.

## 3. The label

`y = 1` if and only if the window's file sits in SWAN-SF's `FL` folder, meaning the largest GOES
flare associated with that active region in the 24 hours after the cutoff was **M- or X-class**.
`y = 0` for the `NF` folder: a B-class, C-class, or no flare.

**The label comes from the folder and file name only.** It is never read from a column that is also
a predictor. The GOES letter of each window (`B`, `C`, `F`, `M`, `X`) is retained in the metadata
for auditing and post-hoc analysis, and is never used at training or prediction time.

## 4. Scale and class balance

| | Count | Share |
|---|---|---|
| Windows | 331,185 | 100 % |
| Positive (M or X) | 6,234 | 1.88 % |
| — M-class | 5,689 | |
| — X-class | 545 | |
| Negative | 324,951 | 98.12 % |
| — B-class followed | 18,125 | |
| — C-class followed | 32,584 | |
| — flare-quiet | 274,242 | |

Time span: **2010-05-01 to 2018-08-17**, covering the rise, peak and decline of solar cycle 24.
Active regions: 3,205 distinct HARP numbers.

## 5. Partitions and the split

| Partition | Windows | Positives | Active regions | First start → last end | Split |
|---|---|---|---|---|---|
| P1 | 73,492 | 1,254 | 700 | 2010-05-01 → 2012-03-13 | train |
| P2 | 88,557 | 1,401 | 885 | 2012-03-02 → 2013-10-28 | train |
| P3 | 42,510 | 1,424 | 390 | 2013-10-17 → 2014-06-14 | train |
| P4 | 51,261 | 1,165 | 472 | 2014-06-02 → 2015-03-18 | **validation** |
| P5 | 75,365 | 990 | 758 | 2015-03-07 → 2018-08-17 | **test (locked)** |

Train = 204,559 windows, validation = 51,261, test = 75,365.

### Two honest notes on the split

**Active regions never overlap.** All ten pairwise intersections of the partitions' HARP sets are
empty, verified in `results/data_audit.json` and asserted in `tests/test_leakage.py`.

**The time spans do overlap at the edges.** P1 ends 2012-03-13 while P2 starts 2012-03-02; P2 ends
2013-10-28 while P3 starts 2013-10-17; and so on for every boundary. This is a property of
SWAN-SF's own partitioning, which groups by active region rather than cutting the calendar cleanly.
The partitions are therefore *region-disjoint and broadly chronological*, **not** strictly
non-overlapping in time. Stated plainly because the stronger claim would be wrong: a region active
in early March 2012 may appear in P1 while a different region observed on the same day appears in
P2. Since no region is shared, no window's near-duplicate crosses the split, which is the property
that matters for leakage.

## 6. Why the split is not random — the overlap statistic

Consecutive windows of one active region are usually **one hour apart**, while each window covers
12 hours. Adjacent windows therefore share **11 of their 12 hours** of observations and are close to
duplicates.

**Definition, stated exactly.** Group the 331,185 windows by `ar`, sort each group by the window's
start timestamp, and difference consecutive timestamps. A group of *k* windows contributes *k* − 1
gaps, so the total is 331,185 − 3,205 = **327,980 gaps**.

**Result: 323,377 of 327,980 gaps (98.5966 %, i.e. 98.60 %) are exactly 1.00 h.**

The statistic is robust to how it is defined:

| Definition | Groups | Gaps | Exactly 1.00 h |
|---|---|---|---|
| by `ar`, ordered by window **start** | 3,205 | 327,980 | **98.5966 %** |
| by `ar`, ordered by window **end** | 3,205 | 327,980 | 98.5966 % |
| by (`partition`, `ar`), ordered by start | 3,205 | 327,980 | 98.5966 % |
| by (`ar`, `folder`), ordered by start | 3,360 | 327,825 | 98.5398 % |
| by `ar`, de-duplicated on (`ar`, `start`) | 3,205 | 327,980 | 98.5966 % |

Minimum gap 1.00 h, maximum 171.00 h, median 1.00 h. Non-one-hour gaps cluster at 12–16 h
(15 h: 748, 14 h: 534, 13 h: 457, 16 h: 333, 12 h: 331) — observation outages, not irregular
sampling. There are no duplicate (`ar`, `start`) pairs and no duplicate file names.

> ### Discrepancy with the report
>
> The written report states **97.9 %**. The recomputation gives **98.60 %** under every definition
> tried, and 97.9 % would require 321,092 one-hour gaps — 2,285 fewer than are present. The report
> is frozen and was not edited; this is recorded here so that whichever figure is published can be
> traced to a definition. Regenerate with `python -m solarflare audit`, which writes
> `results/extra/window_gaps.json`.

A random train/test split would place near-duplicate windows on both sides and inflate every skill
score. The chronological, region-disjoint partition split is what prevents that.

## 7. Predictors: the 144 features

The 24 SHARP parameters observed **inside the window only**:

```
TOTUSJH  TOTBSQ   TOTPOT   TOTUSJZ  ABSNJZH  SAVNCPP  USFLUX  TOTFZ
MEANPOT  EPSZ     MEANSHR  SHRGT45  MEANGAM  MEANGBT  MEANGBZ MEANGBH
MEANJZH  TOTFY    MEANJZD  MEANALP  TOTFX    EPSY     EPSX    R_VALUE
```

× 6 statistics, each computed over that parameter's **finite** values in the window:

| Statistic | Definition |
|---|---|
| `last` | the last finite value — the value at the prediction cutoff. If the final record is missing, the latest value that is not. |
| `mean` | arithmetic mean of the finite values |
| `std` | **population** standard deviation, ddof = 0 |
| `min`, `max` | extremes of the finite values |
| `slope` | least-squares linear trend **per hour**, fitted on the finite points against a time axis of `index × 0.2` h. NaN unless at least two finite points exist spanning more than an instant. |

= **144 features**, named `<PARAM>__<stat>`, ordered parameter-major. A parameter that is entirely
missing yields NaN for all six of its statistics, left to the pipeline's median imputer.

`src/common.py` is the single source for these lists; `tests/test_features.py` pins every edge case
on synthetic windows.

## 8. Deliberately excluded columns

Excluding these is the core of the leakage argument. `tests/test_leakage.py` asserts that the
exclusion set and the predictor set are disjoint.

| Group | Columns | Why excluded |
|---|---|---|
| Label-derived | `BFLARE`, `CFLARE`, `MFLARE`, `XFLARE` and their `_LOC`, `_LABEL`, `_LABEL_LOC` variants | They encode the flare outcome itself. Using any of them would be direct label leakage. |
| GOES X-ray | `XR_MAX`, `XR_QUAL` | Direct measurements of flare activity, overlapping the prediction window. |
| Position / geometry | `CRVAL1`, `CRLN_OBS`, `CRLT_OBS`, `CRVAL2`, `HC_ANGLE`, `LAT_MIN`, `LON_MIN`, `LAT_MAX`, `LON_MAX` | Observational geometry, not magnetic physics. Correlates with time of observation and would let the model learn the calendar. |
| Quality / flags | `QUALITY`, `SPEI`, `IS_TMFI` | Audited, never used to predict and never used to filter the data. |

## 9. Missing data and quality

| Quantity | Value |
|---|---|
| Windows containing at least one missing SHARP value | **10.43 %** |
| Mean per-window missing fraction across the 60 × 24 cells | 0.597 % |
| Missing fraction of the final 331,185 × 144 feature matrix | 0.168 % |
| Mean fraction of records with a non-zero `QUALITY` flag | 4.30 % |

**Quality flags are audited, not used to filter.** No window is dropped for a non-zero `QUALITY`
value. Dropping them would change the evaluation population and make the results incomparable with
the SWAN-SF benchmark.

Missing values are handled by median imputation fitted on P1–P3 only, inside the pipeline.

## 10. Known data-integrity problem with the raw archives

On the machine these results were verified on, **`partition2_instances.tar.gz` and
`partition5_instances.tar.gz` are corrupt gzip streams.** P1, P3 and P4 decompress cleanly; P2 fails
with "invalid block type" and P5 with "invalid distance code". A streamed tar read stops *quietly*
at the damage, so P5 yields only **64,315 of its 75,365** windows.

**No published result is affected.** The per-chunk feature files in `data/features/` were extracted
while the archives were intact — their logged counts are the full 88,557 for P2 and 75,365 for P5 —
and every downstream number reproduces exactly from them.

What it does affect: re-running the extraction from these files would silently produce a short
dataset. `python -m solarflare features --check-archives` now verifies every archive before
extracting and refuses to proceed. Two of the twenty windows in the dashboard's explorer view lack
their raw series for this reason and say so. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## 11. Region identifiers: HARP, not NOAA

SWAN-SF's `ar` field is the **HARP number** (HMI Active Region Patch), not the NOAA active region
number. There is no one-to-one correspondence: one HARP may contain several NOAA regions, and many
HARPs — typically small or spotless ones — have no NOAA counterpart at all.

All new documentation and UI in this project says "HARP". A NOAA number is added only where the
mapping has been checked against JSOC's published `all_harps_with_noaa_ars.txt`:

- **HARP 7115 → NOAA AR 12673** — verified. This is the September 2017 region, the most flare-productive
  of cycle 24, and it supplies the dashboard's positive demo window.
- **HARP 5413 → no NOAA counterpart** — verified absent from the mapping file. It supplies the quiet
  demo window, and is labelled "no NOAA AR assigned" rather than given a number.

## 12. Files

| Path | Contents | In git? |
|---|---|---|
| `raw_data/partition{1-5}_instances.tar.gz` | The SWAN-SF archives, 5.7 GB | no |
| `data/features/P{1-5}_c*_X.npz` + `_meta.csv` | Per-chunk extracted features, 241 MB | no |
| `data/all_X.npz` | Merged 331,185 × 144 float32 matrix, 180 MB | no |
| `data/all_meta.csv.gz` | One row per window: `file, folder, flare_class, goes_letter, y, role, ar, start, end, n_rows, nan_frac, bad_quality_frac, first_ts, last_ts, partition` | no |
| `results/data_audit.json` | The audit summarised in this document | **yes** |
| `tests/fixtures/parity_windows.json.gz` | 220 real windows, for the JavaScript parity test | **yes** |

Rebuild the merged files from the chunks in about four minutes with `python -m solarflare audit`,
which also re-verifies the audit against the frozen copy.
