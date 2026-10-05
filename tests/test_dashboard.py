"""Browser smoke tests for the dashboard.

Every view must render without a console error, and the numbers it shows must
equal the ones in ``results/``. The six original views are checked in report
mode as well, since the report's Figures 5.1-5.6 were captured that way.

Skipped cleanly when Playwright or its Chromium build is unavailable.
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
import sys
import time

import pytest

from solarflare import models, paths
from solarflare.shoot_views import ALL_VIEWS, LEGACY_VIEWS

playwright_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
pytestmark = pytest.mark.playwright

PORT = 8799


def _port_open(port: int) -> bool:
    with contextlib.closing(socket.socket()) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def server():
    """Serve dashboard/ for the duration of the module."""
    if _port_open(PORT):
        yield PORT
        return
    process = subprocess.Popen(
        [sys.executable, "-m", "solarflare", "dashboard", "--port", str(PORT)],
        cwd=paths.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            if _port_open(PORT):
                break
            time.sleep(0.25)
        else:
            process.terminate()
            pytest.skip("the dashboard server did not start")
        yield PORT
    finally:
        process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=10)


@pytest.fixture(scope="module")
def page(server):
    """A loaded dashboard page that records every console error."""
    errors: list[str] = []
    try:
        with playwright_api.sync_playwright() as play:
            try:
                browser = play.chromium.launch()
            except Exception as exc:
                pytest.skip(
                    f"chromium is unavailable: {exc}. "
                    f"Run: python -m playwright install chromium"
                )
            tab = browser.new_page(viewport={"width": 1440, "height": 960})
            tab.on(
                "console",
                lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None,
            )
            tab.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            tab.goto(f"http://localhost:{server}/index.html", wait_until="networkidle")
            tab.wait_for_function("window.dashboardReady === true", timeout=30_000)
            tab.errors = errors
            yield tab
            browser.close()
    except playwright_api.Error as exc:
        pytest.skip(f"playwright could not run: {exc}")


def test_page_loads_without_errors(page):
    assert page.evaluate("window.dashboardReady") is True
    assert page.evaluate("window.dashboardError || null") is None
    assert not page.errors, "console errors: " + "; ".join(page.errors)


@pytest.mark.parametrize("view", ALL_VIEWS)
def test_every_view_renders(page, view):
    page.evaluate(f"window.showView('{view}')")
    page.wait_for_timeout(180)
    assert page.is_visible(f"#v-{view}"), f"{view} did not become visible"
    text = page.inner_text(f"#v-{view}")
    assert len(text.strip()) > 60, f"{view} rendered almost no content"
    assert "—\n" not in text[:40], f"{view} still shows its placeholder dashes"
    assert not page.errors, "console errors: " + "; ".join(page.errors)


def test_only_one_view_is_visible_at_a_time(page):
    page.evaluate("window.showView('eval')")
    visible = page.evaluate(
        "[...document.querySelectorAll('.view')].filter(v=>v.classList.contains('active'))"
        ".map(v=>v.id)"
    )
    assert visible == ["v-eval"]


def test_report_mode_hides_the_new_navigation(page):
    """So the report's Figures 5.1-5.6 stay visually equivalent."""
    page.evaluate("window.setReportMode(true)")
    page.wait_for_timeout(120)
    hidden = page.evaluate(
        "[...document.querySelectorAll('.nav-extra')].every(b=>b.offsetParent===null)"
    )
    assert hidden, "the added nav buttons are still visible in report mode"
    assert page.evaluate("document.getElementById('tourBtn').offsetParent === null")

    legacy_visible = page.evaluate(
        "[...document.querySelectorAll('nav button')]"
        ".filter(b=>b.offsetParent!==null).map(b=>b.dataset.v)"
    )
    assert legacy_visible == list(LEGACY_VIEWS)
    page.evaluate("window.setReportMode(false)")


def test_evaluation_view_numbers_equal_results_json(page):
    """Every KPI on the evaluation view must match results/test_results.json."""
    frozen = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))[models.selected_key()]
    page.evaluate("window.showView('eval')")
    page.wait_for_timeout(150)
    text = page.inner_text("#eval-kpi")
    for key in ("precision", "recall", "f1", "tss", "pr_auc", "roc_auc"):
        assert (
            f"{frozen[key]:.3f}" in text
        ), f"{key} = {frozen[key]:.3f} is not shown on the evaluation view"


def test_confusion_matrix_matches_the_frozen_counts(page):
    frozen = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))[models.selected_key()]
    page.evaluate("window.showView('eval')")
    page.wait_for_timeout(150)
    text = page.inner_text("#cm-grid")
    for key in ("tp", "fp", "fn", "tn"):
        assert f"{frozen[key]:,}" in text, f"{key} = {frozen[key]:,} is not in the matrix"


def test_thresholds_shown_match_selection_json(page):
    alert, _ = models.thresholds()
    page.evaluate("window.showView('result')")
    page.wait_for_timeout(150)
    assert f"{alert:.4f}" in page.inner_text("#v-result")


def test_risk_bands_are_generated_not_hard_coded(page):
    """The bands must come from context.json, at the real threshold values."""
    alert, high = models.thresholds()
    page.evaluate("window.showView('risk')")
    page.wait_for_timeout(150)
    text = page.inner_text("#riskrow")
    assert f"{alert:.2f}" in text
    assert f"{high:.2f}" in text
    for level in ("LOW", "MODERATE", "HIGH"):
        assert level in text


def test_operating_view_resets_to_the_published_operating_point(page):
    """The slider's default must reproduce the frozen confusion matrix exactly."""
    frozen = json.loads(paths.TEST_JSON.read_text(encoding="utf-8"))[models.selected_key()]
    alert, _ = models.thresholds()
    page.evaluate("window.showView('operate')")
    page.wait_for_timeout(200)
    page.click("#op-reset")
    page.wait_for_timeout(200)

    assert page.inner_text("#op-thr") == f"{alert:.4f}"
    matrix = page.inner_text("#op-cm")
    for key in ("tp", "fp", "fn", "tn"):
        assert f"{frozen[key]:,}" in matrix, (
            f"at the published threshold the slider shows a matrix without "
            f"{key} = {frozen[key]:,}"
        )


def test_operating_slider_changes_the_numbers(page):
    page.evaluate("window.showView('operate')")
    page.wait_for_timeout(150)
    before = page.inner_text("#op-kpi")
    page.evaluate(
        "const s=document.getElementById('op-slider');"
        "s.value=Math.max(0, Number(s.value)-40);"
        "s.dispatchEvent(new Event('input'))"
    )
    page.wait_for_timeout(150)
    assert page.inner_text("#op-kpi") != before


def test_operating_view_carries_the_exploratory_warning(page):
    page.evaluate("window.showView('operate')")
    page.wait_for_timeout(120)
    assert "Exploratory" in page.inner_text("#op-banner")


def test_explore_view_lists_windows_and_opens_one(page):
    page.evaluate("window.showView('explore')")
    page.wait_for_timeout(200)
    count = page.evaluate("document.querySelectorAll('.wcard').length")
    assert count >= 15
    page.evaluate("document.querySelector('.wcard').click()")
    page.wait_for_timeout(200)
    detail = page.inner_text("#wdetail")
    assert "HARP" in detail
    assert "contributions" in detail.lower()


def test_explore_hides_the_truth_until_asked(page):
    page.evaluate("window.showView('explore')")
    page.wait_for_timeout(150)
    page.evaluate("document.querySelector('.wcard').click()")
    page.wait_for_timeout(200)
    assert "Reveal the ground truth" in page.inner_text("#rev")
    page.click("#revBtn")
    page.wait_for_timeout(150)
    revealed = page.inner_text("#rev")
    assert "y = 1" in revealed or "y = 0" in revealed


def test_live_inference_runs_the_built_in_example(page):
    page.evaluate("window.showView('live')")
    page.wait_for_timeout(150)
    page.click("#useExample")
    page.wait_for_timeout(400)
    # The labels are uppercased by CSS, so compare case-insensitively.
    result = page.inner_text("#live-result")
    assert "probability of m/x" in result.lower()
    assert any(level in result for level in ("LOW", "MODERATE", "HIGH"))
    assert "60" in page.inner_text("#live-status")
    assert not page.errors, "console errors: " + "; ".join(page.errors)


def test_live_inference_agrees_with_the_stored_probability(page):
    """The browser's own answer must match the frozen prediction for that window."""
    payload = json.loads(paths.DASHBOARD_WINDOWS.read_text(encoding="utf-8"))
    example = next(w for w in payload["windows"] if w["series_available"])

    page.evaluate("window.showView('live')")
    page.click("#useExample")
    page.wait_for_timeout(400)
    shown = page.inner_text("#live-result")
    assert f"{example['probability']:.4f}" in shown, (
        f"the live view shows a probability other than the stored "
        f"{example['probability']:.4f} for {example['file']}"
    )


def test_footer_disclaimer_is_present_on_every_view(page):
    for view in ALL_VIEWS:
        page.evaluate(f"window.showView('{view}')")
        page.wait_for_timeout(80)
        footer = page.inner_text("footer")
        assert "not an operational space-weather warning service" in footer, view


def test_guided_tour_walks_every_step(page):
    page.click("#tourBtn")
    page.wait_for_timeout(200)
    assert page.is_visible("#tour")
    steps = page.evaluate("TOUR.length")
    assert steps >= 5
    seen = set()
    for _ in range(steps):
        seen.add(page.inner_text("#tour-title"))
        page.click("#tour-next")
        page.wait_for_timeout(120)
    assert len(seen) == steps, "the tour repeated a step"
    assert not page.is_visible("#tour"), "the tour did not close on the last step"
    assert not page.errors, "console errors: " + "; ".join(page.errors)


def test_no_console_errors_after_the_whole_walkthrough(page):
    assert not page.errors, "console errors: " + "; ".join(page.errors)
