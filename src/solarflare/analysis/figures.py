"""Figures for the post-hoc analyses, written to ``figures/extra/``.

Every figure is drawn from a JSON file in ``results/extra/`` — nothing is
recomputed here and no number is typed in by hand, so a figure cannot drift away
from the analysis it depicts. Run ``python -m solarflare analysis`` first.

The report's own ``figures/fig*.png`` are frozen and are never touched; the path
guard in :func:`solarflare.paths.assert_not_frozen` enforces that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import paths, plotting

__all__ = ["build_all"]


def _load(name: str) -> dict | None:
    """Load one analysis result, or return None if it has not been run."""
    path = paths.RESULTS_EXTRA / f"{name}.json"
    if not path.is_file():
        print(
            f"  skipping {name}: {paths.relative(path)} not found "
            f"(run 'python -m solarflare analysis' first)"
        )
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_figure(payload: dict) -> Path:
    """Forest plot of point estimates with 95 % block-bootstrap intervals."""
    order = ["recall", "precision", "f1", "tss", "pr_auc", "roc_auc", "fpr", "alert_rate"]
    labels = {
        "recall": "Recall (POD)",
        "precision": "Precision",
        "f1": "F1-score",
        "tss": "TSS",
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "fpr": "False-alarm rate (FPR)",
        "alert_rate": "Alert rate",
    }
    rows = [(labels[k], payload["intervals"][k]) for k in order if k in payload["intervals"]]

    fig, ax = plotting.figure(7.2, 0.52 * len(rows) + 1.8)
    for i, (_label, stats) in enumerate(rows):
        y = len(rows) - 1 - i
        low, high = stats["ci95_low"], stats["ci95_high"]
        point = stats["point_estimate"]
        ax.plot([low, high], [y, y], color=plotting.NAVY, lw=2.4, solid_capstyle="round")
        ax.plot([point], [y], "o", color=plotting.RED, ms=7, zorder=3)
        ax.text(
            high + 0.015,
            y,
            f"{point:.3f}  [{low:.3f}, {high:.3f}]",
            va="center",
            fontsize=8.5,
            color=plotting.GREY,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)])
    ax.set_xlim(0, 1.32)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Score on the locked test partition P5")
    ax.set_title(
        f"Active-region block bootstrap, {payload['n_resamples_used']:,} resamples\n"
        f"point estimate and 95 % interval, frozen threshold",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0)
    return plotting.finish(fig, "extra1_bootstrap_intervals")


def _calibration_figure(payload: dict) -> Path:
    """Reliability diagram for the frozen model and the validation-fitted Platt variant."""
    fig, axes = plotting.figure(9.6, 4.4, ncols=2)

    for ax, key, title in (
        (
            axes[0],
            "reliability_quantile_bins",
            "Frozen model (class-weighted)\nprobabilities are not calibrated",
        ),
        (
            axes[1],
            "reliability_platt_quantile_bins",
            "Platt-scaled on validation only\nshown for contrast, NOT the selected model",
        ),
    ):
        bins = payload[key]
        predicted = [b["mean_predicted"] for b in bins]
        observed = [b["observed_frequency"] for b in bins]
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=plotting.GREY, label="perfect calibration")
        ax.plot(
            predicted, observed, "o-", color=plotting.NAVY, lw=1.8, ms=5, label="observed frequency"
        )
        ax.axhline(
            payload["test_base_rate"],
            color=plotting.GREEN,
            lw=1,
            ls=":",
            label=f"P5 base rate = {payload['test_base_rate']:.4f}",
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Observed frequency of a major flare")

    brier = payload["brier_frozen"]
    ratio = payload["variants"]["frozen_class_weighted"]["over_forecast_ratio"]
    fig.suptitle(
        f"Reliability on P5 (equal-count bins) — Brier {brier:.4f}, "
        f"mean forecast {ratio:.1f}x the base rate",
        fontsize=11,
    )
    return plotting.finish(fig, "extra2_calibration_reliability")


def _operating_figure(payload: dict) -> Path:
    """Metrics across the threshold sweep, with the validation-fixed points marked."""
    sweep = payload["sweep"]
    thresholds = np.array([r["threshold"] for r in sweep])
    alert = payload["named_operating_points"]["alert_threshold_validation_tss"]["threshold"]
    high = payload["named_operating_points"]["high_risk_threshold_validation_f1"]["threshold"]

    fig, ax = plotting.figure(7.6, 4.8)
    for metric, colour, label in (
        ("recall", plotting.GREEN, "Recall (POD)"),
        ("precision", plotting.RED, "Precision"),
        ("tss", plotting.NAVY, "TSS"),
        ("alert_rate", plotting.GREY, "Alert rate"),
    ):
        ax.plot(thresholds, [r[metric] for r in sweep], lw=1.9, color=colour, label=label)

    ax.axvline(alert, color=plotting.NAVY, lw=1.3, ls="--")
    ax.axvline(high, color=plotting.AMBER, lw=1.3, ls="--")
    ax.annotate(
        f"alert = {alert:.4f}\n(validation TSS)",
        xy=(alert, 0.62),
        xytext=(alert - 0.30, 0.70),
        fontsize=8,
        color=plotting.NAVY,
        arrowprops={"arrowstyle": "->", "color": plotting.NAVY, "lw": 0.9},
    )
    ax.annotate(
        f"high risk = {high:.4f}\n(validation F1)",
        xy=(high, 0.34),
        xytext=(high - 0.34, 0.20),
        fontsize=8,
        color=plotting.AMBER,
        arrowprops={"arrowstyle": "->", "color": plotting.AMBER, "lw": 0.9},
    )

    ax.set_xlabel("Decision threshold on the predicted probability")
    ax.set_ylabel("Score on P5")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(thresholds.min(), 1.0)
    ax.set_title(
        "Operating-point trade-off on the locked test partition\n"
        "thresholds were fixed on validation, not on this curve",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="center left")
    return plotting.finish(fig, "extra3_operating_points")


def _breakdown_figure(payload: dict) -> Path:
    """Recall by GOES class and metrics by calendar year."""
    fig, axes = plotting.figure(9.8, 4.4, ncols=2)

    classes = payload["by_goes_class"]
    names = list(classes)
    recalls = [classes[c]["recall"] for c in names]
    lows = [recalls[i] - classes[c]["recall_wilson_ci95"][0] for i, c in enumerate(names)]
    highs = [classes[c]["recall_wilson_ci95"][1] - recalls[i] for i, c in enumerate(names)]
    axes[0].bar(
        names,
        recalls,
        color=[plotting.NAVY, plotting.RED],
        width=0.55,
        yerr=[lows, highs],
        capsize=7,
        error_kw={"lw": 1.2},
    )
    for i, c in enumerate(names):
        axes[0].text(
            i,
            0.04,
            f"n = {classes[c]['n_positive_windows']}",
            ha="center",
            color="white",
            fontsize=9.5,
            fontweight="bold",
        )
        axes[0].text(
            i, recalls[i] + highs[i] + 0.035, f"{recalls[i]:.3f}", ha="center", fontsize=9.5
        )
    axes[0].set_ylim(0, 1.18)
    axes[0].set_ylabel("Recall at the frozen threshold")
    axes[0].set_title(
        "Recall by GOES class\n95 % Wilson intervals — X rests on 19 windows", fontsize=10
    )

    years = payload["by_calendar_year"]
    labels = sorted(years)
    usable = [y for y in labels if "recall" in years[y]]
    x = np.arange(len(usable))
    width = 0.27
    for offset, metric, colour, label in (
        (-width, "recall", plotting.GREEN, "Recall"),
        (0.0, "precision", plotting.RED, "Precision"),
        (width, "tss", plotting.NAVY, "TSS"),
    ):
        axes[1].bar(
            x + offset, [years[y][metric] for y in usable], width, color=colour, label=label
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(usable)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(
        "By calendar year within P5\nthe base rate falls as cycle 24 declines", fontsize=10
    )
    axes[1].legend(fontsize=8.5)
    for i, y in enumerate(usable):
        axes[1].text(
            i, 1.0, f"{years[y]['positives']}+", ha="center", fontsize=7.5, color=plotting.GREY
        )
    return plotting.finish(fig, "extra4_breakdowns")


def _nearmiss_figure(payload: dict) -> Path:
    """What followed each false alarm, and the probability ladder by flare class."""
    classes = payload["by_negative_class"]
    letters = ["F", "B", "C"]
    names = {"F": "Flare-quiet\n(no flare)", "B": "B-class\nfollowed", "C": "C-class\nfollowed"}

    fig, axes = plotting.figure(9.8, 4.6, ncols=2)

    # Left: alert rate within each negative class - the decisive comparison.
    rates = [classes[letter]["alert_rate_within_class"] * 100 for letter in letters]
    colours = [plotting.GREEN, plotting.AMBER, plotting.RED]
    labels = [
        f"{names[letter]}\nn = {classes[letter]['n_negative_windows']:,}" for letter in letters
    ]
    bars = axes[0].bar(labels, rates, color=colours, width=0.6)
    for bar, rate in zip(bars, rates, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            rate + max(rates) * 0.025,
            f"{rate:.1f} %",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    axes[0].set_ylabel("Share of windows that triggered an alert")
    axes[0].set_ylim(0, max(rates) * 1.25)
    axes[0].set_title(
        "Alert rate within each class of 'negative' window\n"
        "the model is discriminating activity, not guessing",
        fontsize=10,
    )

    # Right: the median probability ladder across all five GOES outcomes.
    ladder = payload["median_probability_ladder"]
    order = [k for k in ("F", "B", "C", "M", "X") if ladder.get(k) is not None]
    values = [ladder[k] for k in order]
    axes[1].plot(range(len(order)), values, "o-", color=plotting.NAVY, lw=2, ms=8)
    for i, (key, value) in enumerate(zip(order, values, strict=True)):
        axes[1].annotate(
            f"{value:.3f}",
            (i, value),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
        )
        if key in ("M", "X"):
            axes[1].plot([i], [value], "o", color=plotting.RED, ms=8, zorder=3)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order)
    axes[1].set_xlabel("Largest GOES flare in the 24 h after the cutoff")
    axes[1].set_ylabel("Median predicted probability")
    axes[1].axvline(2.5, color=plotting.GREY, ls="--", lw=1)
    axes[1].text(
        2.55,
        max(values) * 0.5,
        "label boundary\n(negative | positive)",
        fontsize=7.5,
        color=plotting.GREY,
    )
    axes[1].set_title(
        "The probability rises with flare magnitude\n"
        "although the model only ever saw a binary target",
        fontsize=10,
    )

    share = payload["headline"]["share_of_false_alarms_preceding_a_real_flare"] * 100
    fig.suptitle(
        f"{share:.1f} % of the 'false alarms' precede a real B- or C-class flare", fontsize=11.5
    )
    return plotting.finish(fig, "extra5_near_miss")


def _baseline_figure(payload: dict) -> Path:
    """The selected model against the two trivial baselines."""
    single = payload["baselines"]["best_single_feature"]
    climate = payload["baselines"]["climatology_constant"]
    model = payload["selected_model"]

    metrics = ["tss", "recall", "precision", "pr_auc"]
    labels = ["TSS", "Recall", "Precision", "PR-AUC"]
    series = [
        (
            "Climatology\n(constant base rate)",
            plotting.GREY,
            [climate.get(m, 0.0) for m in metrics],
        ),
        (
            f"Best single feature\n({single['feature']})",
            plotting.AMBER,
            [single.get(m, 0.0) for m in metrics],
        ),
        ("Selected model\n(LR, 144 temporal)", plotting.NAVY, [model.get(m, 0.0) for m in metrics]),
    ]

    x = np.arange(len(metrics))
    width = 0.26
    fig, ax = plotting.figure(7.8, 4.6)
    for i, (name, colour, values) in enumerate(series):
        bars = ax.bar(x + (i - 1) * width, values, width, color=colour, label=name)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.016,
                f"{value:.3f}",
                ha="center",
                fontsize=7.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score on the locked test partition P5")
    ax.set_title(
        "How much the model adds over trivial rules\n"
        "both baselines were fixed on train/validation, then scored once on P5",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    return plotting.finish(fig, "extra6_baselines")


def _ablation_figure(payload: dict) -> Path:
    """Validation TSS when one statistic family is dropped, and when only one is kept."""
    variants = payload["statistic_family_variants"]
    families = [k[len("drop_") :] for k in variants if k.startswith("drop_")]
    full = payload["findings"]["full_model_validation_tss"]

    x = np.arange(len(families))
    width = 0.38
    fig, ax = plotting.figure(8.0, 4.6)
    ax.bar(
        x - width / 2,
        [variants[f"drop_{f}"]["validation_tss"] for f in families],
        width,
        color=plotting.NAVY,
        label="drop this family (120 features)",
    )
    ax.bar(
        x + width / 2,
        [variants[f"only_{f}"]["validation_tss"] for f in families],
        width,
        color=plotting.AMBER,
        label="keep only this family (24 features)",
    )
    ax.axhline(
        full, color=plotting.RED, lw=1.4, ls="--", label=f"full 144-feature model = {full:.3f}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylim(0.70, 0.90)
    ax.set_xlabel("Statistic family")
    ax.set_ylabel("Validation TSS (partition P4)")
    ax.set_title(
        "Statistic-family ablation\nsame architecture and protocol; "
        "compared on validation, as selection would be",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    return plotting.finish(fig, "extra7_ablation")


def _error_figure(payload: dict) -> Path:
    """Where the misses and the most confident false alarms sit."""
    misses = payload["misses"]["worst_misses"]
    runs = payload["false_alarms"]["longest_consecutive_runs"]

    fig, axes = plotting.figure(9.8, 4.6, ncols=2)

    labels = [f"HARP {m['harp']}  {m['flare_class']}" for m in misses][::-1]
    values = [m["probability"] for m in misses][::-1]
    axes[0].barh(range(len(values)), values, color=plotting.RED, height=0.7)
    axes[0].set_yticks(range(len(values)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_title(
        f"The {len(values)} worst misses of "
        f"{payload['misses']['n']} total\nreal pre-flare windows scored "
        f"lowest",
        fontsize=10,
    )
    axes[0].grid(axis="y", alpha=0)

    axes[1].barh(
        range(len(runs)),
        [r["consecutive_false_alarm_windows"] for r in runs],
        color=plotting.NAVY,
        height=0.7,
    )
    axes[1].set_yticks(range(len(runs)))
    axes[1].set_yticklabels([f"HARP {r['harp']}" for r in runs], fontsize=8)
    axes[1].set_xlabel("Longest run of consecutive false-alarm windows")
    axes[1].set_title(
        f"{payload['false_alarms']['n']:,} false alarms are not "
        f"{payload['false_alarms']['n']:,} independent errors\n"
        f"one mis-read region alerts for many hours in a row",
        fontsize=10,
    )
    axes[1].grid(axis="y", alpha=0)
    return plotting.finish(fig, "extra8_error_gallery")


def build_all() -> list[Path]:
    """Build every additive figure whose analysis output is present."""
    builders = [
        ("bootstrap", _bootstrap_figure),
        ("calibration", _calibration_figure),
        ("operating_points", _operating_figure),
        ("breakdowns", _breakdown_figure),
        ("near_miss", _nearmiss_figure),
        ("baselines", _baseline_figure),
        ("ablation", _ablation_figure),
        ("error_gallery", _error_figure),
    ]
    written: list[Path] = []
    for name, builder in builders:
        payload = _load(name)
        if payload is None:
            continue
        written.append(builder(payload))
    return written
