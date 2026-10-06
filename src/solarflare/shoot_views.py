"""Capture every dashboard view to ``docs/screenshots/``.

This is the replacement for ``src/shoot.py``, which writes to ``figures/`` and
would overwrite the report's frozen Figures 5.1-5.6. Nothing here touches them.

The six original views are also captured in **report mode**, with the three new
navigation buttons and the tour launcher hidden, so those images stay visually
equivalent to the ones embedded in the finished report.
"""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import paths

__all__ = ["ALL_VIEWS", "LEGACY_FILENAMES", "LEGACY_VIEWS", "capture"]

#: The six views the report's figures were captured from.
LEGACY_VIEWS = ("overview", "predict", "result", "risk", "explain", "eval")
#: Added after the report was frozen. ``start`` is the landing page and the
#: default view; ``replay`` walks a recorded active-region history.
NEW_VIEWS = ("start", "explore", "live", "operate", "replay")
#: Capture order: the landing page first, then the report's six, then the rest.
ALL_VIEWS = ("start",) + LEGACY_VIEWS + ("explore", "live", "operate", "replay")

#: Report figure names, kept so a reviewer can diff against figures/fig5_*.png.
LEGACY_FILENAMES = {
    "overview": "fig5_1_overview",
    "predict": "fig5_2_forecast_config",
    "result": "fig5_3_prediction_result",
    "risk": "fig5_4_risk_alert",
    "explain": "fig5_5_explainability",
    "eval": "fig5_6_model_evaluation",
}

VIEWPORT = {"width": 1440, "height": 960}


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with contextlib.closing(socket.socket()) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


@contextlib.contextmanager
def _server(port: int):
    """Start the dashboard server if nothing is already serving on ``port``."""
    if _port_open(port):
        print(f"  reusing the server already on port {port}")
        yield None
        return
    process = subprocess.Popen(
        [sys.executable, "-m", "solarflare", "dashboard", "--port", str(port)],
        cwd=paths.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            if _port_open(port):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"the dashboard server did not come up on port {port}")
        yield process
    finally:
        process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=10)


def capture(port: int = 8791, out_dir: Path | None = None) -> int:
    """Screenshot every view. Returns a process exit code."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "FAIL - playwright is not installed. Run:\n"
            "  pip install -r requirements-dev.txt\n"
            "  python -m playwright install chromium"
        )
        return 1

    out = out_dir or paths.SCREENSHOTS
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    with _server(port), sync_playwright() as play:
        browser = play.chromium.launch()
        # Capture with reduced motion. The landing page's staggered entrance
        # animation runs for up to 0.62 s, so a fixed wait caught it at a
        # different phase on every run and the screenshot was never twice the
        # same - 10 % of its pixels changed between identical builds. The
        # stylesheet already collapses every animation under this preference,
        # which makes the captures deterministic and exercises that path.
        page = browser.new_page(viewport=VIEWPORT, reduced_motion="reduce")
        page.on(
            "console",
            lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None,
        )
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

        page.goto(f"http://localhost:{port}/index.html", wait_until="networkidle")
        page.wait_for_function("window.dashboardReady === true", timeout=30_000)

        # The six original views, with the new chrome hidden.
        page.evaluate("window.setReportMode(true)")
        for view in LEGACY_VIEWS:
            page.evaluate(f"window.showView('{view}')")
            page.wait_for_timeout(260)
            target = out / f"{LEGACY_FILENAMES[view]}.png"
            paths.assert_not_frozen(target)
            page.screenshot(path=str(target))
            print(f"  wrote {paths.relative(target)}")

        # Every view as the examiner will actually see it.
        page.evaluate("window.setReportMode(false)")
        for i, view in enumerate(ALL_VIEWS, start=1):
            page.evaluate(f"window.showView('{view}')")
            if view == "explore":
                page.evaluate(
                    "document.querySelector('.wcard') && document.querySelector('.wcard').click()"
                )
            if view == "live":
                page.evaluate("document.getElementById('useExample').click()")
            page.wait_for_timeout(400)
            target = out / f"view{i}_{view}.png"
            paths.assert_not_frozen(target)
            page.screenshot(path=str(target), full_page=True)
            print(f"  wrote {paths.relative(target)}")

        browser.close()

    if errors:
        print(f"\nFAIL - {len(errors)} browser error(s):")
        for message in errors[:20]:
            print(f"  {message}")
        return 1
    print(
        f"\n{len(LEGACY_VIEWS) + len(ALL_VIEWS)} screenshots written to "
        f"{paths.relative(out)}; no console errors."
    )
    print("The report's figures/fig5_*.png were not touched.")
    return 0
