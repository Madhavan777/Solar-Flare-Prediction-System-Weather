"""Feature ablation: which part of the representation carries the signal.

Two questions:

1. **Point-in-time vs temporal.** Already answered by the two frozen candidates:
   ``I1_LR_last`` uses the 24 values at the prediction cutoff, the selected
   ``I2_LR_temporal_C0.01`` uses 144 summary statistics over the whole window.
   Their frozen numbers are reproduced here side by side, not recomputed.
2. **Which statistic families matter.** Twelve new logistic regressions are
   trained: six that drop one statistic family (120 features each) and six that
   keep only one family (24 features each).

Every ablation model follows the frozen protocol exactly — fitted on P1–P3,
threshold chosen on P4 by TSS, scored once on P5 — so the comparison is fair.
None of it feeds back into the selection: the frozen model was chosen before any
of this existed, and remains the headline result.
"""

from __future__ import annotations

import sys
import time

from .. import data, paths
from ..metrics import best_threshold, extended
from . import _common

if str(paths.SRC) not in sys.path:
    sys.path.insert(0, str(paths.SRC))
from common import SEED, STATS

__all__ = ["run"]


def _pipeline(C: float = 0.01):
    """The selected model's architecture, rebuilt for a different column subset."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    from common import slog

    return Pipeline(
        [
            ("slog", FunctionTransformer(slog)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED),
            ),
        ]
    )


def run() -> dict:
    """Train the ablation variants, score them, and write the comparison out."""
    import json

    X, meta = data.load()
    train, validation, test = data.split_masks(meta)
    y = meta["y"].to_numpy()
    columns = list(X.columns)

    frozen_val = json.loads(paths.VALIDATION_JSON.read_text(encoding="utf-8"))
    frozen_test = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))

    representation = {
        key: {
            "description": frozen_val[key]["description"],
            "n_features": frozen_val[key]["n_features"],
            "validation_tss": frozen_val[key]["val"]["tss"],
            "validation_pr_auc": frozen_val[key]["val"]["pr_auc"],
            "test_tss": frozen_test[key]["tss"],
            "test_pr_auc": frozen_test[key]["pr_auc"],
            "test_precision": frozen_test[key]["precision"],
            "test_recall": frozen_test[key]["recall"],
            "source": "frozen results/, not recomputed here",
        }
        for key in ("I1_LR_last", "I2_LR_temporal_C0.01")
    }

    variants: dict[str, dict] = {}
    for mode in ("drop", "only"):
        for stat in STATS:
            if mode == "drop":
                subset = [c for c in columns if not c.endswith(f"__{stat}")]
                label = f"drop_{stat}"
                description = (
                    f"144 temporal features minus the {len(columns) - len(subset)} "
                    f"'{stat}' features ({len(subset)} features)"
                )
            else:
                subset = [c for c in columns if c.endswith(f"__{stat}")]
                label = f"only_{stat}"
                description = f"only the {len(subset)} '{stat}' features"

            started = time.time()
            model = _pipeline().fit(X.loc[train, subset], y[train])
            p_val = model.predict_proba(X.loc[validation, subset])[:, 1]
            threshold = best_threshold(y[validation], p_val, "tss")
            p_test = model.predict_proba(X.loc[test, subset])[:, 1]

            val_metrics = extended(y[validation], p_val, threshold)
            test_metrics = extended(y[test], p_test, threshold)
            variants[label] = {
                "description": description,
                "n_features": len(subset),
                "fit_seconds": round(time.time() - started, 1),
                "threshold_chosen_on": "validation partition P4, maximising TSS",
                "threshold": threshold,
                "validation_tss": val_metrics["tss"],
                "validation_pr_auc": val_metrics["pr_auc"],
                "test_tss": test_metrics["tss"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
            print(
                f"  {label:14s} {len(subset):3d} feats  "
                f"val TSS={val_metrics['tss']:.4f}  test TSS={test_metrics['tss']:.4f}  "
                f"test PR-AUC={test_metrics['pr_auc']:.4f}",
                flush=True,
            )

    full_val = frozen_val["I2_LR_temporal_C0.01"]["val"]["tss"]
    full_pr = frozen_test["I2_LR_temporal_C0.01"]["pr_auc"]
    full_test_tss = frozen_test["I2_LR_temporal_C0.01"]["tss"]
    dropped = {k: v for k, v in variants.items() if k.startswith("drop_")}
    most_damaging = min(dropped, key=lambda k: dropped[k]["validation_tss"])
    only = {k: v for k, v in variants.items() if k.startswith("only_")}
    best_single_family = max(only, key=lambda k: only[k]["validation_tss"])

    # The honest headline of this ablation: several smaller variants score HIGHER
    # validation TSS than the selected model. They were not in the pre-registered
    # candidate set, so they could not have been selected - but that is a real
    # limitation of the search, and it must not be buried.
    beating = sorted(
        (
            {
                "variant": name,
                "n_features": stats["n_features"],
                "validation_tss": stats["validation_tss"],
                "validation_tss_above_selected": stats["validation_tss"] - full_val,
                "test_tss": stats["test_tss"],
                "test_pr_auc": stats["test_pr_auc"],
                "test_pr_auc_vs_selected": stats["test_pr_auc"] - full_pr,
            }
            for name, stats in variants.items()
            if stats["validation_tss"] > full_val
        ),
        key=lambda row: -row["validation_tss"],
    )

    payload = {
        "analysis": "feature ablation",
        "protocol": (
            "Every variant is a new logistic regression with the selected model's "
            "architecture (signed-log, median imputation, standardisation, C=0.01, "
            "class_weight='balanced', seed 42), fitted on P1-P3, with its threshold "
            "chosen on P4 by TSS and scored once on P5. The frozen model is unchanged "
            "and no ablation result was used to select anything."
        ),
        "representation_comparison": representation,
        "statistic_family_variants": variants,
        "findings": {
            "full_model_validation_tss": full_val,
            "full_model_test_pr_auc": full_pr,
            "full_model_test_tss": full_test_tss,
            "variants_beating_the_selected_model_on_validation": beating,
            "limitation_this_exposes": (
                (
                    f"{len(beating)} of the {len(variants)} ablation variants reach a higher "
                    f"validation TSS than the selected model. They were never candidates - "
                    f"the seven-candidate set was fixed before any of this was computed, and "
                    f"the selection rule was applied to that set honestly - but it shows the "
                    f"candidate search was narrow. A wider search over feature subsets, run "
                    f"under the same protocol, would very likely have selected a smaller "
                    f"model. Stated as a limitation rather than as a correction: the frozen "
                    f"result stands, and this is what an examiner should be told about it."
                )
                if beating
                else ("No ablation variant beats the selected model on validation TSS.")
            ),
            "most_damaging_family_to_drop": {
                "variant": most_damaging,
                "validation_tss": dropped[most_damaging]["validation_tss"],
                "validation_tss_loss_vs_full": float(
                    full_val - dropped[most_damaging]["validation_tss"]
                ),
            },
            "best_single_family": {
                "variant": best_single_family,
                "validation_tss": only[best_single_family]["validation_tss"],
                "test_pr_auc": only[best_single_family]["test_pr_auc"],
            },
            "max_validation_tss_loss_from_dropping_one_family": float(
                full_val - min(v["validation_tss"] for v in dropped.values())
            ),
        },
        "n_variants_trained": len(variants),
    }
    _common.write("ablation", payload)
    return payload
