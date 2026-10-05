"""What actually followed each false alarm.

The headline precision of 16.2 % treats every alerted negative window as an
equal mistake. SWAN-SF's own labels say otherwise: a negative window is one
whose largest following flare was B-class, C-class, or nothing at all, and those
are very different outcomes.

This analysis asks a single question — **of the windows the model alerted on but
which were labelled negative, what really happened in the next 24 hours?** — and
reports the answer by GOES class.

It changes no published number. It changes how the published precision should be
read, and it is computed entirely from the dataset's own ``goes_letter`` field,
which is derived from the SWAN-SF folder and file names and was never a
predictor.
"""

from __future__ import annotations

import itertools

from .. import models
from . import _common

__all__ = ["run"]

#: Negative-window classes, from quietest to most active.
NEGATIVE_CLASSES = ("F", "B", "C")
CLASS_MEANING = {
    "F": "flare-quiet: no B-, C-, M- or X-class flare in the next 24 h",
    "B": "a B-class flare followed, the weakest GOES category",
    "C": "a C-class flare followed, one order of magnitude below the M threshold",
}


def run() -> dict:
    """Break the false alarms down by what actually followed, and write it out."""
    frame = _common.load_test_frame()
    threshold, _high_threshold = models.thresholds()

    negatives = frame[frame["y"] == 0]
    false_alarms = negatives[negatives["outcome"] == "FP"]
    correct_quiet = negatives[negatives["outcome"] == "TN"]
    positives = frame[frame["y"] == 1]

    by_class = {}
    for letter in NEGATIVE_CLASSES:
        in_class = negatives[negatives["goes_letter"] == letter]
        alarms_in_class = int((false_alarms["goes_letter"] == letter).sum())
        base_share = float((negatives["goes_letter"] == letter).mean())
        alarm_share = float((false_alarms["goes_letter"] == letter).mean())
        by_class[letter] = {
            "meaning": CLASS_MEANING[letter],
            "n_negative_windows": len(in_class),
            "share_of_all_negatives": base_share,
            "n_false_alarms": alarms_in_class,
            "share_of_false_alarms": alarm_share,
            "enrichment_vs_base": float(alarm_share / base_share) if base_share else None,
            "alert_rate_within_class": float(in_class["alert"].mean()) if len(in_class) else None,
            "mean_probability": float(in_class["p"].mean()) if len(in_class) else None,
            "median_probability": float(in_class["p"].median()) if len(in_class) else None,
            "n_reaching_high_risk_band": int((in_class["risk"] == "HIGH").sum()),
        }

    bc_alarms = int(false_alarms["goes_letter"].isin(["B", "C"]).sum())
    c_alarms = int((false_alarms["goes_letter"] == "C").sum())
    quiet_alarms = int((false_alarms["goes_letter"] == "F").sum())

    # Does the model order windows by flare magnitude, as the argument claims?
    ladder = {}
    for letter in ("F", "B", "C"):
        subset = negatives[negatives["goes_letter"] == letter]
        ladder[letter] = float(subset["p"].median()) if len(subset) else None
    for letter in ("M", "X"):
        subset = positives[positives["goes_letter"] == letter]
        ladder[letter] = float(subset["p"].median()) if len(subset) else None
    ordered = [ladder[k] for k in ("F", "B", "C", "M", "X") if ladder[k] is not None]
    monotone = all(a < b for a, b in itertools.pairwise(ordered))

    # A "any flare at all" relabelling: how would precision read if the task were
    # "will this region flare at all?" rather than "will it produce M or X?".
    any_flare = frame["goes_letter"].isin(["B", "C", "M", "X"]).to_numpy()
    alerted = frame["alert"].to_numpy()
    tp_any = int((alerted & any_flare).sum())
    fp_any = int((alerted & ~any_flare).sum())
    precision_any = tp_any / (tp_any + fp_any) if tp_any + fp_any else 0.0

    payload = {
        "analysis": "what actually followed each false alarm",
        "question": (
            "The published precision of 0.1616 counts every alerted negative window as an "
            "equal error. SWAN-SF labels each negative window with the largest flare that "
            "followed it - B, C, or none. This asks what that distribution looks like "
            "among the false alarms."
        ),
        "alert_threshold": threshold,
        "counts": {
            "negative_windows": len(negatives),
            "false_alarms": len(false_alarms),
            "correct_quiet_forecasts": len(correct_quiet),
        },
        "headline": {
            "false_alarms_preceding_a_real_b_or_c_flare": bc_alarms,
            "share_of_false_alarms_preceding_a_real_flare": float(bc_alarms / len(false_alarms)),
            "false_alarms_preceding_a_c_class_flare": c_alarms,
            "share_preceding_a_c_class_flare": float(c_alarms / len(false_alarms)),
            "false_alarms_on_genuinely_flare_quiet_windows": quiet_alarms,
            "share_on_genuinely_flare_quiet_windows": float(quiet_alarms / len(false_alarms)),
        },
        "by_negative_class": by_class,
        "median_probability_ladder": ladder,
        "probability_increases_monotonically_with_flare_class": bool(monotone),
        "relabelled_as_any_flare": {
            "description": (
                "Hypothetical only, and NOT the project's task: if the target were 'will "
                "any B, C, M or X flare follow?' rather than 'will an M or X flare "
                "follow?', the same alerts at the same threshold would score this "
                "precision. Reported to size the effect, not as a result - the model was "
                "never trained or selected for this target, and the frozen metrics stand."
            ),
            "true_positives": tp_any,
            "false_positives": fp_any,
            "precision": float(precision_any),
            "precision_on_the_real_mx_task": float(
                len(frame[frame["outcome"] == "TP"]) / int(alerted.sum())
            ),
        },
        "interpretation": (
            f"{bc_alarms:,} of the {len(false_alarms):,} false alarms "
            f"({bc_alarms / len(false_alarms) * 100:.1f} %) precede a genuine B- or "
            f"C-class flare, and only {quiet_alarms:,} "
            f"({quiet_alarms / len(false_alarms) * 100:.1f} %) land on a window that was "
            f"truly flare-quiet. Within the flare-quiet windows the model alerts only "
            f"{by_class['F']['alert_rate_within_class'] * 100:.2f} % of the time, against "
            f"{by_class['C']['alert_rate_within_class'] * 100:.1f} % within C-class "
            f"windows. The model is therefore discriminating magnetic activity "
            f"successfully and failing mainly to resolve the M-class boundary, which is a "
            f"materially different weakness from alerting at random on quiet regions."
        ),
        "what_this_does_not_excuse": (
            "A C-class flare is not an M-class flare, and an operator told 'major flare "
            "likely' when a C-class flare follows has still been misinformed. The "
            "published precision of 0.1616 for the M/X task is correct and is not revised "
            "by this analysis. What this shows is that the errors are concentrated near "
            "the decision boundary of flare magnitude rather than scattered over quiet "
            "Sun, which is the difference between a miscalibrated threshold and a model "
            "that has learnt nothing."
        ),
    }
    _common.write("near_miss", payload)
    return payload
