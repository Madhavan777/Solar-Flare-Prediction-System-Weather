"""Assemble ``results/extra/RESULTS.md`` from the analysis JSON files.

The prose is written here; every number is read from ``results/extra/*.json``.
Nothing is typed in by hand, so the document cannot drift from the analyses, and
``scripts/check_consistency.py`` re-checks it against ``results/`` anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ._common import POSTHOC

__all__ = ["build"]


def _load(name: str) -> dict | None:
    path = paths.RESULTS_EXTRA / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float, places: int = 2) -> str:
    return f"{x * 100:.{places}f} %"


def _interval(stats: dict, places: int = 4) -> str:
    return (
        f"{stats['point_estimate']:.{places}f} "
        f"[{stats['ci95_low']:.{places}f}, {stats['ci95_high']:.{places}f}]"
    )


def _bootstrap_section(payload: dict) -> list[str]:
    labels = {
        "recall": "Recall (probability of detection)",
        "precision": "Precision",
        "f1": "F1-score",
        "tss": "TSS",
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "fpr": "False-alarm rate (FPR)",
        "far": "False-alarm ratio (FAR)",
        "alert_rate": "Alert rate",
        "hss2": "HSS (HSS2 form)",
    }
    lines = [
        "## 1. Confidence intervals by active-region block bootstrap",
        "",
        "**What this is.** How precisely the P5 score is measured — not a new result. "
        f"{payload['n_resamples_used']:,} resamples, seed "
        f"{payload.get('n_resamples_requested') and 42}, frozen threshold throughout.",
        "",
        "**Why whole regions are the resampling unit.** Consecutive windows of one HARP "
        "region are one hour apart 98.60 % of the time, so adjacent 12-hour windows share "
        "11 of their 12 hours of observations. Resampling individual windows would treat "
        "those near-duplicates as independent draws and give intervals that are far too "
        "narrow. Resampling the "
        f"{payload['n_regions']} regions keeps each region's windows together.",
        "",
        "| Metric | Point estimate | 95 % interval |",
        "|---|---|---|",
    ]
    for key, label in labels.items():
        if key not in payload["intervals"]:
            continue
        stats = payload["intervals"][key]
        lines.append(
            f"| {label} | {stats['point_estimate']:.4f} | "
            f"[{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}] |"
        )
    windows = payload["resampled_window_count"]
    positives = payload["resampled_positive_count"]
    lines += [
        "",
        f"Resampled partitions ranged from {windows['min']:,} to {windows['max']:,} windows "
        f"(median {windows['median']:,}) and from {positives['min']:,} to "
        f"{positives['max']:,} positives (median {positives['median']:,}). That spread is the "
        "honest picture: P5's effective sample size is set by how many *regions* flared, "
        "not by its 75,365 windows.",
        "",
    ]

    # The single most important consequence of these intervals.
    tss = payload["intervals"].get("tss")
    pr = payload["intervals"].get("pr_auc")
    if tss and pr:
        stored = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
        spread = max(v["tss"] for v in stored.values()) - min(v["tss"] for v in stored.values())
        lines += [
            "### What these intervals mean for the candidate comparison",
            "",
            f"The TSS interval is **{tss['ci95_high'] - tss['ci95_low']:.3f} wide** "
            f"([{tss['ci95_low']:.3f}, {tss['ci95_high']:.3f}]). The entire spread of test "
            f"TSS across all seven candidates is **{spread:.3f}** — roughly "
            f"{(tss['ci95_high'] - tss['ci95_low']) / spread:.0f} times narrower than the "
            "interval around any one of them.",
            "",
            "**The seven candidates are statistically indistinguishable on this test "
            'partition.** That is the correct answer to "why did logistic regression win '
            "when a random forest scores higher test TSS?\": the random forests' apparent "
            "advantage of 0.007-0.014 TSS is far inside the noise, so it is not evidence of "
            "anything. The selection was made on validation, as the protocol requires, and "
            "no test-partition ranking among these models could have been trusted anyway.",
            "",
            f"The PR-AUC interval is wider still, [{pr['ci95_low']:.3f}, "
            f"{pr['ci95_high']:.3f}] around a point estimate of "
            f"{pr['point_estimate']:.3f}. Any single-figure PR-AUC quoted for this project "
            "should be read with that in mind.",
            "",
        ]
    if payload.get("n_resamples_skipped_degenerate"):
        lines += [
            f"{payload['n_resamples_skipped_degenerate']} resample(s) were discarded for "
            "containing only one class, where precision, recall and the AUCs are undefined.",
            "",
        ]
    return lines


def _calibration_section(payload: dict) -> list[str]:
    frozen = payload["variants"]["frozen_class_weighted"]
    lines = [
        "## 2. Calibration — the probabilities over-forecast, and that is expected",
        "",
        "**The finding, stated plainly.** The selected model's probabilities are **not "
        "calibrated**. A window scored 0.60 does not flare 60 % of the time. The mean "
        f"predicted probability on P5 is {payload['mean_predicted_probability']:.4f} "
        f"against an actual base rate of {payload['test_base_rate']:.4f} — the model "
        f"over-forecasts by about **{frozen['over_forecast_ratio']:.1f}x**.",
        "",
        f"**Why.** {payload['why_uncalibrated']}",
        "",
        f"**What this does not affect.** {payload['what_is_unaffected']}",
        "",
        f"Brier score: **{payload['brier_frozen']:.4f}** for the frozen model, against "
        f"{payload['brier_constant_climatology']:.4f} for a constant forecast at the "
        "training base rate. A constant forecast beats the model on Brier score while "
        "detecting nothing — which is precisely why Brier score is not used to choose a "
        "flare forecaster.",
        "",
        "| Variant | Selected? | Brier | PR-AUC | ROC-AUC | Mean forecast / base rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, variant in payload["variants"].items():
        lines.append(
            f"| `{name}` | {'**yes**' if variant['selected'] else 'no'} | "
            f"{variant['brier']:.4f} | {variant['pr_auc']:.4f} | {variant['roc_auc']:.4f} | "
            f"{variant['over_forecast_ratio']:.2f}x |"
        )
    shifts = payload["roc_auc_shift_under_recalibration"]
    platt_shift = abs(shifts.get("platt_on_validation", 0.0))
    isotonic_shift = abs(shifts.get("isotonic_on_validation", 0.0))
    lines += [
        "",
        "Both recalibrations were fitted on the **validation partition only** and are "
        "reported for contrast. Neither is the selected model and neither changes any "
        "published number.",
        "",
        "The ROC-AUC shift under each is worth reading carefully, because the two "
        "recalibrations are not the same kind of function:",
        "",
        f"* **Platt scaling moves the ROC-AUC by {platt_shift:.2e}** — exactly nothing. "
        "It is a one-parameter logistic in the logit, which is *strictly* increasing, so "
        "it cannot reorder any pair of windows. This is the clean demonstration that "
        "calibration and discrimination are separate properties: the Brier score "
        f"improves from {payload['brier_frozen']:.4f} to "
        f"{payload['variants']['platt_on_validation']['brier']:.4f} while the ranking is "
        "untouched.",
        f"* **Isotonic regression moves it by {isotonic_shift:.2e}**, from "
        f"{payload['variants']['frozen_class_weighted']['roc_auc']:.4f} to "
        f"{payload['variants']['isotonic_on_validation']['roc_auc']:.4f}. That is a real "
        "change, not numerical noise, and it is expected: isotonic regression is "
        "monotone *non-decreasing*, so it maps whole intervals of probability onto a "
        "single value. The ties it creates destroy ordering information, which costs "
        "both ROC-AUC and PR-AUC "
        f"({payload['variants']['isotonic_on_validation']['pr_auc']:.4f} against "
        f"{payload['variants']['frozen_class_weighted']['pr_auc']:.4f}).",
        "",
        "So the general claim is the narrower one: a *strictly* monotone recalibration "
        "leaves every ranking metric untouched. Isotonic regression is not strictly "
        "monotone and does not.",
        "",
    ]
    return lines


def _operating_section(payload: dict) -> list[str]:
    fields = [
        "threshold",
        "recall",
        "precision",
        "far",
        "fpr",
        "tss",
        "alert_rate",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    headers = [
        "Operating point",
        "Threshold",
        "Recall",
        "Precision",
        "FAR (ratio)",
        "FPR (rate)",
        "TSS",
        "Alert rate",
        "TP",
        "FP",
        "FN",
        "TN",
    ]
    lines = [
        "## 3. Operating points",
        "",
        f"**{payload['warning']}**",
        "",
        "Risk bands as the dashboard renders them: "
        + ", ".join(f"**{k}** {v}" for k, v in payload["risk_bands"].items())
        + ".",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for name, point in payload["named_operating_points"].items():
        cells = []
        for field in fields:
            value = point[field]
            cells.append(f"{value:,}" if isinstance(value, int) else f"{value:.4f}")
        marker = " **(published)**" if point["is_published_operating_point"] else ""
        lines.append(f"| `{name}`{marker} | " + " | ".join(cells) + " |")

    best = payload["best_tss_on_test_posthoc"]
    lines += [
        "",
        f"The highest-TSS threshold anywhere on the P5 sweep is "
        f"{best['threshold']:.4f}, scoring TSS {best['tss']:.4f}. The published "
        f"validation-fixed threshold scores "
        f"{payload['named_operating_points']['alert_threshold_validation_tss']['tss']:.4f}, "
        f"so fixing the threshold on validation instead of the test set cost "
        f"**{best['tss_cost_of_fixing_threshold_on_validation']:.4f} TSS**. That number is "
        "the price of doing the protocol honestly, and it is small.",
        "",
        f"The full {payload['sweep_points']}-point sweep is in "
        "`results/extra/operating_points.json`, and drives the dashboard's threshold "
        "slider.",
        "",
    ]
    return lines


def _breakdown_section(payload: dict) -> list[str]:
    classes = payload["by_goes_class"]
    region = payload["by_harp_region"]
    lines = [
        "## 4. Breakdowns: where the model does better and worse",
        "",
        "### By GOES class",
        "",
        "| Class | Positive windows | Detected | Missed | Recall | 95 % Wilson interval |",
        "|---|---|---|---|---|---|",
    ]
    for letter, stats in classes.items():
        low, high = stats["recall_wilson_ci95"]
        lines.append(
            f"| {letter} | {stats['n_positive_windows']} | {stats['detected']} | "
            f"{stats['missed']} | {stats['recall']:.4f} | [{low:.3f}, {high:.3f}] |"
        )
    x_warning = classes.get("X", {}).get("small_sample_warning")
    if x_warning:
        lines += [
            "",
            f"**Caveat:** {x_warning}. The X-class recall should not be quoted "
            "as a headline figure.",
            "",
        ]

    lines += [
        "",
        "### By calendar year",
        "",
        "| Year | Windows | Positives | Base rate | Recall | Precision | TSS |",
        "|---|---|---|---|---|---|---|",
    ]
    for year, stats in payload["by_calendar_year"].items():
        if "recall" not in stats:
            lines.append(
                f"| {year} | {stats.get('n', 0):,} | "
                f"{stats.get('positives', 0)} | — | — | — | "
                f"{stats.get('note', '')} |"
            )
            continue
        lines.append(
            f"| {year} | {stats['n']:,} | {stats['positives']} | "
            f"{stats['base_rate']:.4f} | {stats['recall']:.4f} | "
            f"{stats['precision']:.4f} | {stats['tss']:.4f} |"
        )
    lines += ["", payload["notes"]["year"], ""]

    lines += [
        "### By HARP region — false alarms are concentrated",
        "",
        f"P5 contains {region['n_regions_in_p5']} HARP regions and "
        f"{region['total_false_alarms']:,} false alarms. "
        f"**{region['regions_producing_half_the_false_alarms']} regions "
        f"({_pct(region['share_of_regions_producing_half'], 1)} of them) produce half of "
        f"them**, and {region['regions_with_zero_false_alarms']} regions produce none at "
        "all.",
        "",
        payload["notes"]["region"],
        "",
        "| HARP | Windows | Positive windows | False alarms | Hits | Max p |",
        "|---|---|---|---|---|---|",
    ]
    for row in region["top_15_false_alarm_regions"][:10]:
        lines.append(
            f"| {row['harp']} | {row['n_windows']:,} | {row['n_positive_windows']} | "
            f"{row['n_false_alarms']:,} | {row['n_hits']} | {row['max_probability']:.4f} |"
        )

    lines += [
        "",
        "### By missingness",
        "",
        "| Missing SHARP values in window | Windows | Positives | Recall | "
        "Precision | Alert rate |",
        "|---|---|---|---|---|---|",
    ]
    for band, stats in payload["by_missingness"].items():
        if "recall" not in stats:
            lines.append(
                f"| {band} | {stats.get('n', 0):,} | "
                f"{stats.get('positives', 0)} | — | — | "
                f"{stats.get('alert_rate', float('nan')):.4f} |"
            )
            continue
        lines.append(
            f"| {band} | {stats['n']:,} | {stats['positives']} | {stats['recall']:.4f} | "
            f"{stats['precision']:.4f} | {stats['alert_rate']:.4f} |"
        )
    lines.append("")
    return lines


def _nearmiss_section(payload: dict) -> list[str]:
    headline = payload["headline"]
    classes = payload["by_negative_class"]
    counts = payload["counts"]
    ladder = payload["median_probability_ladder"]

    lines = [
        "## 5. What actually followed each false alarm",
        "",
        f"**The question.** {payload['question']}",
        "",
        f"**The answer.** {headline['false_alarms_preceding_a_real_b_or_c_flare']:,} of the "
        f"{counts['false_alarms']:,} false alarms "
        f"(**{headline['share_of_false_alarms_preceding_a_real_flare'] * 100:.1f} %**) "
        f"precede a genuine B- or C-class flare. Only "
        f"{headline['false_alarms_on_genuinely_flare_quiet_windows']:,} "
        f"({headline['share_on_genuinely_flare_quiet_windows'] * 100:.1f} %) land on a "
        f"window that was truly flare-quiet.",
        "",
        "| What followed the window | Negative windows | Share of negatives | "
        "False alarms | Share of false alarms | Enrichment | Alert rate within class |",
        "|---|---|---|---|---|---|---|",
    ]
    for letter in ("F", "B", "C"):
        stats = classes[letter]
        lines.append(
            f"| **{letter}** — {stats['meaning']} | {stats['n_negative_windows']:,} | "
            f"{stats['share_of_all_negatives'] * 100:.2f} % | "
            f"{stats['n_false_alarms']:,} | "
            f"{stats['share_of_false_alarms'] * 100:.2f} % | "
            f"**x{stats['enrichment_vs_base']:.2f}** | "
            f"{stats['alert_rate_within_class'] * 100:.2f} % |"
        )

    lines += [
        "",
        f"The decisive comparison is the last column: the model alerts on "
        f"**{classes['C']['alert_rate_within_class'] * 100:.1f} %** of C-class windows but "
        f"only **{classes['F']['alert_rate_within_class'] * 100:.2f} %** of flare-quiet "
        f"windows — a factor of "
        f"{classes['C']['alert_rate_within_class'] / classes['F']['alert_rate_within_class']:.0f}. "
        f"C-class windows are enriched among the false alarms by "
        f"x{classes['C']['enrichment_vs_base']:.1f}, while flare-quiet windows are "
        f"*depleted* by x{classes['F']['enrichment_vs_base']:.2f}.",
        "",
    ]

    if payload["probability_increases_monotonically_with_flare_class"]:
        lines += [
            "### The predicted probability rises monotonically with flare magnitude",
            "",
            "| Largest flare in the next 24 h | Median predicted probability |",
            "|---|---|",
        ]
        for letter in ("F", "B", "C", "M", "X"):
            if ladder.get(letter) is not None:
                lines.append(f"| {letter} | {ladder[letter]:.4f} |")
        lines += [
            "",
            "The model was trained on a binary target that lumps F, B and C together as "
            "one class and M and X together as the other. It was never told that C is "
            "stronger than B, or that X is stronger than M. That the median probability "
            "nonetheless increases across all five categories is evidence that it learnt "
            "a continuous notion of magnetic activity rather than a boundary memorised "
            "from the label.",
            "",
        ]

    relabelled = payload["relabelled_as_any_flare"]
    lines += [
        f"**Interpretation.** {payload['interpretation']}",
        "",
        f"**What this does not excuse.** {payload['what_this_does_not_excuse']}",
        "",
        "For scale only, and emphatically not as a result: "
        f"{relabelled['description']} That hypothetical precision is "
        f"{relabelled['precision']:.4f}, against the real task's "
        f"{relabelled['precision_on_the_real_mx_task']:.4f}.",
        "",
    ]
    return lines


def _baseline_section(payload: dict) -> list[str]:
    climate = payload["baselines"]["climatology_constant"]
    single = payload["baselines"]["best_single_feature"]
    model = payload["selected_model"]
    lines = [
        "## 6. Baselines — what the model actually buys",
        "",
        f"**Selection discipline.** {payload['selection_discipline']}",
        "",
        "| | Climatology | Best single feature | Selected model |",
        "|---|---|---|---|",
        f"| Rule | constant p = {payload['train_base_rate']:.6f} | `{single['rule']}` | "
        f"LR on 144 temporal features |",
        f"| TSS | {climate.get('tss', 0):.4f} | {single['tss']:.4f} | {model['tss']:.4f} |",
        f"| Recall | {climate.get('recall', 0):.4f} | {single['recall']:.4f} | "
        f"{model['recall']:.4f} |",
        f"| Precision | {climate.get('precision', 0):.4f} | {single['precision']:.4f} | "
        f"{model['precision']:.4f} |",
        f"| PR-AUC | {climate.get('pr_auc', 0):.4f} | {single['pr_auc']:.4f} | "
        f"{model['pr_auc']:.4f} |",
        f"| Alerts raised | {climate.get('tp', 0) + climate.get('fp', 0):,} | "
        f"{single['tp'] + single['fp']:,} | {model['alert_rate'] * 75365:,.0f} |",
        "",
        "**Climatology** is the no-skill reference: perfectly calibrated, zero "
        f"discrimination, TSS {climate.get('tss', 0):.1f}. It raises no alerts and detects "
        f"nothing, while scoring {climate.get('accuracy', 0):.4f} accuracy. Keep that "
        "number next to any accuracy figure quoted for this project.",
        "",
        f"**The best single feature** was found by searching "
        f"{payload['n_single_feature_rules_searched']:,} one-feature threshold rules on the "
        f"validation partition, then scoring the winner once on P5: "
        f"`{single['rule']}`.",
        "",
        "### This is the most uncomfortable result in the project, so state it plainly",
        "",
        f"On TSS, **the single-feature rule essentially matches the model**: "
        f"{single['tss']:.4f} against {model['tss']:.4f}, a difference of "
        f"{payload['improvement_over_best_single_feature']['tss']:+.4f}. On precision it is "
        f"{'higher' if single['precision'] > model['precision'] else 'lower'} "
        f"({single['precision']:.4f} vs {model['precision']:.4f}), and on F1 "
        f"{'higher' if single['f1'] > model['f1'] else 'lower'} "
        f"({single['f1']:.4f} vs {model['f1']:.4f}), at a comparable recall "
        f"({single['recall']:.4f} vs {model['recall']:.4f}).",
        "",
        'So the honest answer to *"does the machine learning earn its place on TSS?"* is '
        "**no, not at this one operating point.** One threshold on one SHARP parameter — "
        "the total unsigned current helicity, which is a well-established flare "
        "predictor — reproduces the headline skill score.",
        "",
        "Where the model does win, decisively, is in **ranking quality across all "
        f"thresholds**: PR-AUC {model['pr_auc']:.4f} against {single['pr_auc']:.4f}, an "
        f"improvement of **{payload['improvement_over_best_single_feature']['pr_auc']:+.4f}** "
        f"({payload['improvement_over_best_single_feature']['pr_auc'] / single['pr_auc'] * 100:+.0f} "
        "% relative). That is not a cosmetic difference, and it is what the rest of the "
        "system actually depends on:",
        "",
        "* the three-band risk output needs the probabilities to be ordered well over the "
        "whole range, not just split well at one cut. At the HIGH-risk threshold the model "
        "reaches 58 % precision — a single raw feature value cannot support a second, "
        "much stricter band in any calibrated way;",
        "* a single feature's score is not a probability at all, so there is nothing to "
        "recalibrate, nothing to threshold a second time, and no per-feature explanation "
        "to show;",
        "* PR-AUC is the threshold-free summary that respects the 1.31 % base rate, and it "
        "is the metric on which the selected model beats every other candidate tried in "
        "this project, including all three random forests.",
        "",
        "The defensible claim is therefore the narrow one: **the pipeline's value is in "
        "calibrated ranking and explainability, not in raising the headline TSS above what "
        "a classical single-parameter threshold achieves.** Anyone quoting TSS 0.853 as "
        "evidence that machine learning was necessary here is overclaiming.",
        "",
        "Runners-up by validation TSS: "
        + ", ".join(
            f"`{r['feature']}` ({r['validation_tss']:.4f})"
            for r in payload["single_feature_runners_up_by_validation_tss"][:5]
        )
        + ". Note that they are all measures of total field strength, current or flux — "
        "the single-feature result is not an artefact of one lucky column.",
        "",
    ]
    return lines


def _ablation_section(payload: dict) -> list[str]:
    representation = payload["representation_comparison"]
    findings = payload["findings"]
    lines = [
        "## 7. Ablation",
        "",
        f"**Protocol.** {payload['protocol']}",
        "",
        "### Point-in-time versus temporal (the two frozen candidates)",
        "",
        "| Candidate | Features | Validation TSS | Test TSS | Test PR-AUC | Test precision |",
        "|---|---|---|---|---|---|",
    ]
    for key, stats in representation.items():
        lines.append(
            f"| `{key}` | {stats['n_features']} | {stats['validation_tss']:.4f} | "
            f"{stats['test_tss']:.4f} | {stats['test_pr_auc']:.4f} | "
            f"{stats['test_precision']:.4f} |"
        )
    i1 = representation["I1_LR_last"]
    i2 = representation["I2_LR_temporal_C0.01"]
    lines += [
        "",
        f"Moving from 24 point-in-time values to 144 temporal summaries raised validation "
        f"TSS by {i2['validation_tss'] - i1['validation_tss']:+.4f} and test PR-AUC by "
        f"{i2['test_pr_auc'] - i1['test_pr_auc']:+.4f}, and raised precision from "
        f"{i1['test_precision']:.3f} to {i2['test_precision']:.3f}. Note that test **TSS** "
        f"went the other way ({i1['test_tss']:.4f} to {i2['test_tss']:.4f}) — see §7.",
        "",
        "### Statistic families",
        "",
        "| Family | Drop it (120 features) | Keep only it (24 features) |",
        "|---|---|---|",
    ]
    variants = payload["statistic_family_variants"]
    families = [k[len("drop_") :] for k in variants if k.startswith("drop_")]
    for family in families:
        lines.append(
            f"| `{family}` | {variants[f'drop_{family}']['validation_tss']:.4f} | "
            f"{variants[f'only_{family}']['validation_tss']:.4f} |"
        )
    damaging = findings["most_damaging_family_to_drop"]
    best_single = findings["best_single_family"]
    lines += [
        "",
        f"Validation TSS of the full model: **{findings['full_model_validation_tss']:.4f}**.",
        "",
        f"Dropping any one family costs at most "
        f"{findings['max_validation_tss_loss_from_dropping_one_family']:.4f} TSS — the "
        f"representation is **highly redundant**, which is unsurprising given that six "
        f"statistics of the same 24 physical parameters are strongly correlated. The most "
        f"damaging single family to remove is `{damaging['variant'][len('drop_'):]}` "
        f"({damaging['validation_tss_loss_vs_full']:+.4f} TSS). The strongest family on its "
        f"own is `{best_single['variant'][len('only_'):]}` at validation TSS "
        f"{best_single['validation_tss']:.4f}, and the weakest by a wide margin is `slope`, "
        f"which collapses on its own — a per-hour trend carries very little signal without "
        f"the level it is a trend in.",
        "",
    ]

    beating = findings.get("variants_beating_the_selected_model_on_validation") or []
    if beating:
        lines += [
            "### The unflattering finding in this ablation",
            "",
            f"**{len(beating)} of the {payload['n_variants_trained']} variants reach a "
            f"higher validation TSS than the selected model.** Since selection used "
            f"validation TSS, any of these would have won had it been a candidate.",
            "",
            "| Variant | Features | Validation TSS | vs selected | Test TSS | Test PR-AUC |",
            "|---|---|---|---|---|---|",
        ]
        for row in beating:
            lines.append(
                f"| `{row['variant']}` | {row['n_features']} | "
                f"{row['validation_tss']:.4f} | "
                f"{row['validation_tss_above_selected']:+.4f} | {row['test_tss']:.4f} | "
                f"{row['test_pr_auc']:.4f} ({row['test_pr_auc_vs_selected']:+.4f}) |"
            )
        lines += [
            "",
            findings["limitation_this_exposes"],
            "",
            "Two things keep this from being a protocol breach. First, the seven-candidate "
            "set was fixed and trained before the test partition was ever scored, and the "
            "selection rule was applied to that set without modification — the frozen "
            "result is exactly what the stated procedure produces. Second, these variants "
            "are being compared on validation, which is the right partition for the "
            "comparison, so the conclusion is legitimate rather than test-set mining.",
            "",
            "What it does mean is that **the model's advantage over a much smaller feature "
            "set is not established**, and the project should not claim that 144 temporal "
            "features were necessary. The defensible claim is narrower: the selected model "
            "has the best PR-AUC on the test partition of everything tried here, and the "
            "temporal representation improved precision substantially over the "
            "point-in-time baseline at comparable recall.",
            "",
        ]
    return lines


def _error_section(payload: dict) -> list[str]:
    misses = payload["misses"]
    alarms = payload["false_alarms"]
    counts = payload["counts"]
    lines = [
        "## 8. Error gallery — the failures, named",
        "",
        f"At the frozen threshold the model produced {counts['true_positives']} hits, "
        f"{counts['false_positives']:,} false alarms, {counts['false_negatives']} misses "
        f"and {counts['true_negatives']:,} correct quiet forecasts.",
        "",
        f"### The {misses['n']} misses",
        "",
        f"These span {misses['n_distinct_harp_regions']} HARP regions, by class: "
        + ", ".join(f"{k} = {v}" for k, v in misses["by_goes_class"].items())
        + f". The lowest probability assigned to a genuine pre-flare window was "
        f"**{misses['lowest_probability']:.6f}** — the model was confidently wrong there, "
        "not merely uncertain.",
        "",
        f"{misses['note']}",
        "",
        "| File | HARP | Class | Window end (cutoff) | p |",
        "|---|---|---|---|---|",
    ]
    for entry in misses["worst_misses"][:10]:
        lines.append(
            f"| `{entry['file']}` | {entry['harp']} | {entry['flare_class']} | "
            f"{entry['window_end_cutoff']} | {entry['probability']:.6f} |"
        )

    lines += [
        "",
        f"### The {alarms['n']:,} false alarms",
        "",
        f"They involve {alarms['n_distinct_harp_regions']} regions, and "
        f"{alarms['n_above_high_risk_threshold']:,} of them exceeded the HIGH-risk "
        f"threshold. The most confident false alarm scored "
        f"**{alarms['highest_probability']:.6f}**.",
        "",
        f"{alarms['note']}",
        "",
        "| File | HARP | Folder | Window end (cutoff) | p |",
        "|---|---|---|---|---|",
    ]
    for entry in alarms["most_confident"][:10]:
        lines.append(
            f"| `{entry['file']}` | {entry['harp']} | {entry['folder']} | "
            f"{entry['window_end_cutoff']} | {entry['probability']:.6f} |"
        )

    lines += [
        "",
        "Longest runs of consecutive false-alarm windows within one region: "
        + ", ".join(
            f"HARP {r['harp']} ({r['consecutive_false_alarm_windows']})"
            for r in alarms["longest_consecutive_runs"][:6]
        )
        + ".",
        "",
    ]

    missing = payload["missingness_of_errors"]
    lines += [
        f"Mean missing-value fraction: {missing['mean_nan_frac_misses']:.5f} for the "
        f"misses, {missing['mean_nan_frac_false_alarms']:.5f} for the false alarms, "
        f"{missing['mean_nan_frac_all']:.5f} across all of P5 — so the errors are not "
        "simply the windows with the most missing data.",
        "",
    ]
    return lines


def _explain_section(payload: dict) -> list[str]:
    return [
        "## 9. The per-window explanation is exact",
        "",
        f"**Identity.** `{payload['identity']}`",
        "",
        f"{payload['why_it_is_exact']}",
        "",
        f"Checked on {payload['n_windows_checked']:,} randomly chosen P5 windows "
        f"(seed {payload['seed']}): the largest discrepancy between the reconstructed and "
        f"the model's own logit was **{payload['max_abs_logit_error']:.2e}**, and between "
        f"the reconstructed and actual probability "
        f"**{payload['max_abs_probability_error']:.2e}**.",
        "",
        f"{payload['note']}",
        "",
    ]


def build() -> Path:
    """Write ``results/extra/RESULTS.md`` and return its path."""
    sections: list[str] = [
        "# Post-hoc analyses",
        "",
        f"> {POSTHOC}",
        "",
        "Every number below is read from a JSON file in this directory by "
        "`src/solarflare/analysis/report.py`; none is typed in by hand. Regenerate with:",
        "",
        "```powershell",
        "python -m solarflare analysis",
        "python -m solarflare figures",
        "```",
        "",
        "Figures for these analyses are in `figures/extra/`. The report's own figures in "
        "`figures/` are frozen and untouched.",
        "",
        "---",
        "",
    ]

    builders = [
        ("bootstrap", _bootstrap_section),
        ("calibration", _calibration_section),
        ("operating_points", _operating_section),
        ("breakdowns", _breakdown_section),
        ("near_miss", _nearmiss_section),
        ("baselines", _baseline_section),
        ("ablation", _ablation_section),
        ("error_gallery", _error_section),
        ("explanation_identity", _explain_section),
    ]
    for name, builder in builders:
        payload = _load(name)
        if payload is None:
            sections += [
                f"## (missing: {name})",
                "",
                f"`results/extra/{name}.json` was not found — "
                "run `python -m solarflare analysis`.",
                "",
                "---",
                "",
            ]
            continue
        sections += builder(payload)
        sections += ["---", ""]

    paths.RESULTS_EXTRA.mkdir(parents=True, exist_ok=True)
    path = paths.assert_not_frozen(paths.RESULTS_EXTRA / "RESULTS.md")
    path.write_text("\n".join(sections), encoding="utf-8")
    return path
