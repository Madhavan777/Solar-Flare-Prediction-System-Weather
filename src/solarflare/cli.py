"""Cross-platform command line entry point.

Everything the project can do is reachable as ``python -m solarflare <command>``,
so the Makefile is a convenience rather than a requirement and the same commands
work in PowerShell and in a POSIX shell.

Run ``python -m solarflare --help`` for the list, or see README.md.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import env, paths

PORT = 8791

#: How far a from-scratch retrain may land from the frozen result and still
#: count as reproducing the decision. Set from the measured drift between the
#: original Linux run and a Windows retrain: the alert threshold moved by
#: 1.2e-3 and the test TSS by 1.7e-3. See cmd_train's docstring.
RETRAIN_TOLERANCE = 5e-3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _heading(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def _write_json(path: Path, payload: object) -> Path:
    """Write JSON to an additive output path, refusing frozen targets."""
    paths.assert_not_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"wrote {paths.relative(path)}")
    return path


def _link_dir(source: Path, target: Path) -> str:
    """Make ``target`` refer to ``source``'s contents as cheaply as possible.

    Tries a symlink, then a Windows directory junction, then a full copy. Used by
    the clean-room retrain so that 190 MB of features need not be duplicated.

    Returns:
        Which mechanism was used: ``"symlink"``, ``"junction"`` or ``"copy"``.
    """
    try:
        # os.symlink, not Path.symlink_to: the target_is_directory flag is
        # required on Windows and Path.symlink_to does not expose it usefully here.
        os.symlink(source, target, target_is_directory=True)  # noqa: PTH211
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0:
            return "junction"
    shutil.copytree(source, target)
    return "copy"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_env(args: argparse.Namespace) -> int:
    """Print the runtime environment and whether it can load the models."""
    _heading("environment")
    print(f"project root: {paths.ROOT}")
    print(env.format_report())
    try:
        env.check(strict=args.strict)
    except env.EnvironmentMismatch as exc:
        print(f"\nFAIL\n{exc}")
        return 1
    print("\nOK — the saved pipelines can be loaded.")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Extract features from the raw SWAN-SF tarballs (slow)."""
    _heading("feature extraction from raw tarballs")
    if not paths.RAW_DATA.is_dir():
        print(
            f"FAIL — {paths.relative(paths.RAW_DATA)} not found. Download SWAN-SF v1.2 "
            f"(DOI 10.7910/DVN/EBCFKM) into that directory first."
        )
        return 1
    from . import data

    if args.check_archives or args.force:
        print(
            "verifying the raw archives are complete gzip streams "
            "(this reads ~5.7 GB and takes a few minutes)..."
        )
        status = data.check_raw_archives()
        bad = {p: s for p, s in status.items() if s != "ok"}
        for partition, state in status.items():
            print(f"  partition{partition}: {state}")
        if bad:
            print(
                "\nFAIL - one or more archives is corrupt or missing. Re-extracting "
                "from them would silently produce a SHORT dataset, because a streamed "
                "tar read stops quietly at the damage. Re-download SWAN-SF v1.2 "
                "(DOI 10.7910/DVN/EBCFKM) before running the extraction."
            )
            return 1
        if args.check_archives:
            return 0

    existing = sorted(paths.FEATURE_CHUNKS.glob("P*_c*_X.npz"))
    if existing and not args.force:
        print(
            f"{len(existing)} feature chunks already exist in "
            f"{paths.relative(paths.FEATURE_CHUNKS)}; nothing to do. Pass --force to redo "
            f"the extraction (hours — the tarballs are streamed sequentially)."
        )
        return 0
    paths.FEATURE_CHUNKS.mkdir(parents=True, exist_ok=True)
    script = paths.SRC / "extract_features.py"
    partitions = [args.partition] if args.partition else [1, 2, 3, 4, 5]
    for partition in partitions:
        tarball = paths.RAW_DATA / f"partition{partition}_instances.tar.gz"
        if not tarball.is_file():
            print(f"FAIL — {paths.relative(tarball)} not found")
            return 1
        start, chunk, index = 0, args.chunk_size, 0
        while True:
            prefix = paths.FEATURE_CHUNKS / f"P{partition}_c{index}"
            print(f"\n-> P{partition} chunk {index} (rows {start}..{start + chunk})", flush=True)
            done = subprocess.run(
                [sys.executable, str(script), str(tarball), str(prefix), str(start), str(chunk)],
                cwd=paths.ROOT,
                check=False,
            )
            if done.returncode != 0:
                return done.returncode
            if prefix.with_name(prefix.name + "_EMPTY").exists():
                break
            if not (prefix.parent / f"{prefix.name}_X.npz").exists():
                break
            import numpy as np

            produced = len(np.load(prefix.parent / f"{prefix.name}_X.npz")["X"])
            if produced < chunk:
                break
            start += chunk
            index += 1
    print("\nfeature extraction finished. Run 'python -m solarflare audit' to merge.")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Merge the feature chunks if needed, then verify the data audit."""
    from . import data, verify

    _heading("data merge and audit")
    if args.force or not (paths.ALL_X.exists() and paths.ALL_META.exists()):
        data.merge_chunks(force=args.force)
    else:
        print(
            f"{paths.relative(paths.ALL_X)} and {paths.relative(paths.ALL_META)} "
            f"already present; skipping the merge (pass --force to rebuild)."
        )

    X, meta = data.load()
    report = verify.verify_audit(X, meta)
    print(report.summary())

    paths.ensure_output_dirs()
    _write_json(paths.RESULTS_EXTRA / "data_audit_recomputed.json", report.details["audit"])

    gaps = {
        "note": "Definition of the consecutive-window gap statistic. See "
        "docs/DATA_CARD.md. Post-hoc description of the data, not a model result.",
        "definitions": {
            "by_ar_start": data.window_gap_stats(meta, by="ar", time="start"),
            "by_ar_end": data.window_gap_stats(meta, by="ar", time="end"),
        },
    }
    _write_json(paths.RESULTS_EXTRA / "window_gaps.json", gaps)
    primary = gaps["definitions"]["by_ar_start"]
    print(
        f"\nconsecutive same-HARP windows exactly 1 h apart: "
        f"{primary['frac_exactly_1h'] * 100:.4f} % "
        f"({primary['n_gaps_exactly_1h']:,} of {primary['n_gaps']:,} gaps)"
    )
    return 0 if report.ok else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Recompute every frozen number from the saved models and compare."""
    from . import verify

    _heading("reproduction of the frozen results")
    env.check()
    report = verify.verify_all()
    print(report.summary())
    paths.ensure_output_dirs()
    _write_json(
        paths.RESULTS_EXTRA / "verification.json",
        {
            "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": env.report(),
            "checks": report.checks,
            "passed": report.ok,
            "failures": report.failures,
            "output": report.lines,
        },
    )
    return 0 if report.ok else 1


def cmd_train(args: argparse.Namespace) -> int:
    """Retrain all seven candidates into a scratch directory and compare.

    Never writes into ``models/`` or ``results/``: the originals stay frozen.
    Logistic regressions take about 15-30 s each, random forests about 4 min
    each, so budget roughly 15 minutes.

    **What this does and does not check.** It checks that the *decision*
    reproduces: that the selection rule picks the same model, and that the
    thresholds and headline metrics land within :data:`RETRAIN_TOLERANCE`. It
    does **not** require bit-identical coefficients.

    Measured on the verification machine, a Windows / Python 3.12 retrain moved
    the alert threshold by 1.2e-3 and the test TSS by 1.8e-3 relative to the
    original Linux / Python 3.11 run, shifting two windows across the decision
    boundary. The selected model and the selection rule were unchanged.

    **The cause is not the thread count.** Fitting the same architecture on the
    same data with 1 and with 4 BLAS threads gives bit-identical coefficients,
    the same intercept and the same iteration count, so floating-point
    accumulation order under parallelism is ruled out. The remaining
    difference is elsewhere in the numerical environment — most plausibly the
    SciPy version, since lbfgs lives in ``scipy.optimize`` and SciPy was the one
    dependency the original specification did not pin, and/or the platform's
    BLAS build. That cannot be settled without the original environment, so it
    is recorded as unresolved rather than asserted.

    Bit-level reproducibility of the published numbers is established by
    ``evaluate`` instead, which scores the *shipped* pipelines and matches to
    1e-9. Pass ``--strict`` to demand exact equality here as well.
    """
    _heading("clean-room retrain (into a scratch directory)")
    out = Path(args.out).resolve() if args.out else paths.ROOT / "build" / "retrain"

    if args.compare_only:
        if not (out / "results" / "selection.json").is_file():
            print(f"FAIL - no retrain output in {out}. Run without --compare-only first.")
            return 1
        print(f"comparing the existing retrain output in {out}; nothing retrained.")
    else:
        if out.exists() and not args.force:
            print(
                f"FAIL - {out} already exists. Pass --force to replace it, or "
                f"--compare-only to re-check it without retraining."
            )
            return 1
        if out.exists():
            shutil.rmtree(out)
        for sub in ("models", "results"):
            (out / sub).mkdir(parents=True, exist_ok=True)

        mechanism = _link_dir(paths.DATA, out / "data")
        print(f"data/ made available in the scratch directory via {mechanism}")

        environment = dict(os.environ, PYTHONPATH=str(paths.SRC))
        started = time.time()
        done = subprocess.run(
            [sys.executable, str(paths.SRC / "train_eval.py")],
            cwd=out,
            env=environment,
            check=False,
        )
        print(f"\ntrain_eval.py exited {done.returncode} after {time.time() - started:.0f}s")
        if done.returncode != 0:
            return done.returncode

    new_selection = json.loads((out / "results" / "selection.json").read_text(encoding="utf-8"))
    old_selection = json.loads(paths.SELECTION_JSON.read_text(encoding="utf-8"))
    tolerance = 0.0 if args.strict else RETRAIN_TOLERANCE
    problems: list[str] = []
    drifts: list[str] = []

    print("\nselection decision (must match exactly):")
    for field_name in ("selected", "rule"):
        new, old = new_selection[field_name], old_selection[field_name]
        print(f"  {field_name:18s} {'OK  ' if new == old else 'DIFF'}  {new!r}")
        if new != old:
            problems.append(f"{field_name}: retrained {new!r} vs frozen {old!r}")

    print(f"\nthresholds (tolerance {tolerance:g}):")
    for field_name in ("alert_threshold", "high_threshold"):
        new, old = float(new_selection[field_name]), float(old_selection[field_name])
        delta = abs(new - old)
        ok = delta <= tolerance
        print(f"  {field_name:18s} {'OK  ' if ok else 'DIFF'}  {new!r}  (delta {delta:.2e})")
        if not ok:
            problems.append(f"{field_name}: retrained {new!r} vs frozen {old!r}, delta {delta:.2e}")
        elif delta > 0:
            drifts.append(f"{field_name} moved by {delta:.2e}")

    # Headline test metrics of the selected model, which is what a reader cares
    # about far more than the coefficients.
    new_test_path = out / "results" / "test_results.json"
    if new_test_path.is_file():
        new_test = json.loads(new_test_path.read_text(encoding="utf-8"))
        old_test = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))
        key = old_selection["selected"]
        print(f"\nheadline test metrics for {key} (tolerance {tolerance:g}):")
        for metric in ("tss", "pr_auc", "roc_auc", "precision", "recall"):
            new, old = float(new_test[key][metric]), float(old_test[key][metric])
            delta = abs(new - old)
            ok = delta <= tolerance
            print(
                f"  {metric:18s} {'OK  ' if ok else 'DIFF'}  {new:.6f}  "
                f"(frozen {old:.6f}, delta {delta:.2e})"
            )
            if not ok:
                problems.append(f"test {metric}: delta {delta:.2e} exceeds {tolerance:g}")
            elif delta > 0:
                drifts.append(f"test {metric} moved by {delta:.2e}")
        print(
            f"  confusion          retrained "
            f"{new_test[key]['tp']}/{new_test[key]['fp']}/{new_test[key]['fn']}/"
            f"{new_test[key]['tn']}  frozen "
            f"{old_test[key]['tp']}/{old_test[key]['fp']}/{old_test[key]['fn']}/"
            f"{old_test[key]['tn']}"
        )

    if problems:
        print(
            f"\nFAIL - a retrain with seed 42 did not reproduce the decision within "
            f"{tolerance:g}:"
        )
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nPASS - a from-scratch retrain reproduces the selection decision.")
    if drifts:
        print("\nNumerical drift, within tolerance and expected across environments:")
        for drift in drifts:
            print(f"  - {drift}")
        print(
            "\n  A retrain converges to slightly different coefficients than the frozen\n"
            "  model. This is NOT the thread count: fitting with 1 and with 4 BLAS threads\n"
            "  gives bit-identical coefficients. The remaining difference is elsewhere in\n"
            "  the numerical environment between the original run (Linux, Python 3.11) and\n"
            "  this one - most plausibly the SciPy version, since lbfgs lives in\n"
            "  scipy.optimize and SciPy was the one dependency the original specification\n"
            "  did not pin. Unresolved, and recorded as such.\n"
            "\n  The published numbers are reproduced bit-for-bit from the SHIPPED pipelines\n"
            "  by 'solarflare evaluate'. This command checks that the DECISION is robust.\n"
            "  Re-run with --strict to demand exact equality."
        )
    print(f"\nScratch output left in {out} (frozen artefacts untouched).")
    return 0


def cmd_analysis(args: argparse.Namespace) -> int:
    """Run the post-hoc analyses into results/extra/."""
    from .analysis import run_all

    _heading("post-hoc analyses (not used for model selection)")
    env.check()
    paths.ensure_output_dirs()
    return run_all(resamples=args.resamples, only=args.only)


def cmd_figures(args: argparse.Namespace) -> int:
    """Build the additive figures into figures/extra/."""
    from .analysis import figures as analysis_figures

    _heading("additive figures (figures/extra/)")
    env.check()
    paths.ensure_output_dirs()
    written = analysis_figures.build_all()
    for path in written:
        print(f"wrote {paths.relative(path)}")
    print(f"\n{len(written)} figures written. " f"The report's figures/fig*.png were not touched.")
    return 0


def cmd_demo_data(args: argparse.Namespace) -> int:
    """Regenerate every JSON the dashboard reads."""
    from .dashboard_data import build_all

    _heading("dashboard data")
    env.check()
    written = build_all(n_windows=args.windows)
    for path in written:
        print(f"wrote {paths.relative(path)} ({path.stat().st_size:,} B)")
    return 0


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already accepting connections on ``port``."""
    import socket

    with contextlib.closing(socket.socket()) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Serve dashboard/ over HTTP on localhost."""
    import http.server

    _heading(f"serving the dashboard on http://localhost:{args.port}/index.html")

    # index.html and demo.json are fatal: without them there is no page worth
    # serving. The rest only disable a view each, so the server starts and says
    # which views will be degraded - it used to start silently and let the
    # browser be the first to notice.
    required = (paths.DASHBOARD / "index.html", paths.DASHBOARD_DEMO)
    optional = {
        paths.DASHBOARD_CONTEXT: "the landing page",
        paths.DASHBOARD_MODEL_LR: "the live view",
        paths.DASHBOARD_EXAMPLE: "the live view's built-in example",
        paths.DASHBOARD_WINDOWS: "the explorer",
        paths.DASHBOARD_OPERATING: "the operating-point view",
        paths.DASHBOARD_REPLAY: "the replay view",
    }
    missing = [p for p in required if not p.exists()]
    if missing:
        print("FAIL - missing: " + ", ".join(paths.relative(m) for m in missing))
        print("  Regenerate with: python -m solarflare demo-data")
        return 1

    degraded = {p: what for p, what in optional.items() if not p.exists()}
    if degraded:
        print(f"WARN - {len(degraded)} data file(s) missing; those views will explain themselves:")
        for p, what in degraded.items():
            print(f"    {paths.relative(p):<32} {what}")
        print("  Regenerate with: python -m solarflare demo-data")

    # On Windows, SO_REUSEADDR lets a second process bind a port that is already
    # being served, so running this command twice silently leaves two servers
    # fighting over one port with no error at all. Refuse up front instead.
    if port_in_use(args.port):
        print(
            f"FAIL - something is already serving on port {args.port}.\n"
            f"  If it is this dashboard, just open "
            f"http://localhost:{args.port}/index.html - there is nothing to start.\n"
            f"  To run a second copy alongside it, choose another port:\n"
            f"      python -m solarflare dashboard --port {args.port + 1}"
        )
        return 1

    directory = str(paths.DASHBOARD)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, fmt, *a):  # quieter output
            if args.verbose:
                super().log_message(fmt, *a)

    # Threading matters: a single-threaded server is blocked for everyone by one
    # keep-alive connection, so a second tab - or a screenshot run while the page
    # is already open - hangs instead of loading.
    class Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        print(
            f"FAIL - could not bind port {args.port}: {exc}\n"
            f"  Try another port:  python -m solarflare dashboard --port {args.port + 1}"
        )
        return 1

    with server as httpd:
        print("press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def cmd_screenshots(args: argparse.Namespace) -> int:
    """Capture every dashboard view into docs/screenshots/."""
    from .shoot_views import capture

    _heading("dashboard screenshots -> docs/screenshots/")
    return capture(port=args.port)


def cmd_test(args: argparse.Namespace) -> int:
    """Run the pytest suite."""
    _heading("pytest")
    command = [sys.executable, "-m", "pytest"]
    if args.fast:
        command += ["-m", "not slow and not playwright"]
    if args.verbose:
        command += ["-v"]
    command += args.pytest_args
    return subprocess.run(command, cwd=paths.ROOT, check=False).returncode


def cmd_check(args: argparse.Namespace) -> int:
    """Verify that every number in the docs and dashboard matches results/."""
    _heading("documentation / dashboard number consistency")
    script = paths.ROOT / "scripts" / "check_consistency.py"
    if not script.is_file():
        print(f"FAIL — {paths.relative(script)} not found")
        return 1
    return subprocess.run([sys.executable, str(script)], cwd=paths.ROOT, check=False).returncode


def cmd_all(args: argparse.Namespace) -> int:
    """Run the whole verification and build pipeline in order."""
    steps = [
        ("environment", cmd_env, argparse.Namespace(strict=False)),
        ("data audit", cmd_audit, argparse.Namespace(force=False)),
        ("reproduction", cmd_evaluate, argparse.Namespace()),
        (
            "post-hoc analyses",
            cmd_analysis,
            argparse.Namespace(resamples=args.resamples, only=None),
        ),
        ("additive figures", cmd_figures, argparse.Namespace()),
        ("dashboard data", cmd_demo_data, argparse.Namespace(windows=20)),
        ("consistency check", cmd_check, argparse.Namespace()),
        ("tests", cmd_test, argparse.Namespace(fast=args.fast, verbose=False, pytest_args=[])),
    ]
    results: list[tuple[str, int]] = []
    for name, function, namespace in steps:
        code = function(namespace)
        results.append((name, code))
        if code != 0 and not args.keep_going:
            _heading("ABORTED")
            print(
                f"'{name}' failed with exit code {code}. "
                f"Fix it before continuing, or re-run with --keep-going."
            )
            return code

    _heading("summary")
    width = max(len(n) for n, _ in results)
    for name, code in results:
        print(f"  {name:<{width}}  {'PASS' if code == 0 else f'FAIL ({code})'}")
    failed = [n for n, c in results if c != 0]
    if failed:
        print(f"\n{len(failed)} step(s) failed: {', '.join(failed)}")
        return 1
    print("\nALL GREEN.")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for every command."""
    parser = argparse.ArgumentParser(
        prog="python -m solarflare",
        description="Solar Flare Prediction & Space Weather Alert System — "
        "academic prototype, not an operational warning service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The models in models/ and every file in results/ are frozen. No command "
        "writes to them; additive output goes to results/extra/, figures/extra/ "
        "and docs/screenshots/.",
    )
    parser.add_argument("--version", action="version", version="solarflare 1.0.0")
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="show the full stack trace instead of a one-line message on expected errors",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("env", help="report the runtime environment")
    p.add_argument(
        "--strict", action="store_true", help="fail on any version deviation, not just scikit-learn"
    )
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("features", help="extract features from raw_data/ (hours)")
    p.add_argument("--partition", type=int, choices=[1, 2, 3, 4, 5], help="only this partition")
    p.add_argument("--chunk-size", type=int, default=35000, help="windows per chunk")
    p.add_argument("--force", action="store_true", help="re-extract over existing chunks")
    p.add_argument(
        "--check-archives",
        action="store_true",
        help="only verify the raw tarballs are complete gzip streams, then exit",
    )
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("audit", help="merge feature chunks and verify the data audit")
    p.add_argument("--force", action="store_true", help="rebuild the merged files")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("evaluate", help="recompute all frozen metrics and compare")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="clean-room retrain into a scratch directory")
    p.add_argument("--out", help="scratch directory (default: build/retrain)")
    p.add_argument("--force", action="store_true", help="replace an existing scratch directory")
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="re-check an existing retrain without training again",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="demand bit-identical thresholds and metrics, not just the "
        "same decision (expected to fail across numerical environments)",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("analysis", help="run the post-hoc analyses")
    p.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples (default 1000)")
    p.add_argument("--only", help="comma-separated subset of analyses to run")
    p.set_defaults(func=cmd_analysis)

    p = sub.add_parser("figures", help="build figures/extra/")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("demo-data", help="regenerate the dashboard JSON files")
    p.add_argument(
        "--windows", type=int, default=20, help="windows in the explorer view (default 20)"
    )
    p.set_defaults(func=cmd_demo_data)

    p = sub.add_parser("dashboard", help="serve dashboard/ on localhost")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("screenshots", help="capture dashboard views to docs/screenshots/")
    p.add_argument("--port", type=int, default=PORT)
    p.set_defaults(func=cmd_screenshots)

    p = sub.add_parser("test", help="run the pytest suite")
    p.add_argument("--fast", action="store_true", help="skip slow and browser tests")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("pytest_args", nargs="*", help="extra arguments passed to pytest")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("check", help="check documented numbers against results/")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("all", help="run everything in order")
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--fast", action="store_true", help="skip slow and browser tests")
    p.add_argument("--keep-going", action="store_true", help="do not stop at the first failure")
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code."""
    # The Windows console defaults to a legacy code page, which mangles any
    # non-ASCII character this tool prints. Ask for UTF-8 and degrade quietly.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2

    # A missing input or a wrong library version is an ordinary, expected
    # condition with an actionable message. Printing a traceback for it buries
    # that message and looks like a crash, which is the last thing wanted when
    # someone is running this in front of an audience. Pass --traceback for the
    # full stack when actually debugging.
    try:
        return int(args.func(args))
    except (FileNotFoundError, env.EnvironmentMismatch, PermissionError) as exc:
        if getattr(args, "traceback", False):
            raise
        print(f"\nFAIL - {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
