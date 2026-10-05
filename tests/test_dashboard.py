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


# --------------------------------------------------------------------------- #
# regressions found by auditing the live page
# --------------------------------------------------------------------------- #
def test_run_forecast_button_actually_does_something(page):
    """It used to be a primary-styled button with no handler at all."""
    page.evaluate("window.showView('predict')")
    page.wait_for_timeout(150)
    assert page.is_visible("#runForecast")
    page.click("#runForecast")
    page.wait_for_timeout(300)
    assert page.is_visible("#v-result"), "Run Forecast did not open the Prediction view"
    gauge = page.inner_text("#g-pct")
    assert gauge.endswith("%") and gauge != "—", f"the gauge still reads {gauge!r}"
    assert page.inner_text("#r-class").strip() in ("M/X flare likely", "No major flare expected")


def test_live_contribution_card_is_hidden_until_an_inference_runs(page):
    """A duplicate style attribute meant the hide never applied.

    HTML keeps the first `style` and drops the second, so an empty
    "Why - exact contributions" card was visible from page load.
    """
    page.reload(wait_until="networkidle")
    page.wait_for_function("window.dashboardReady === true", timeout=30_000)
    page.evaluate("window.showView('live')")
    page.wait_for_timeout(150)
    assert (
        page.evaluate("getComputedStyle(document.getElementById('live-contrib-card')).display")
        == "none"
    )

    page.click("#useExample")
    page.wait_for_timeout(400)
    assert (
        page.evaluate("getComputedStyle(document.getElementById('live-contrib-card')).display")
        != "none"
    )
    assert page.inner_text("#live-contrib").strip()


def test_choosing_a_window_brings_the_detail_into_view(page):
    """Clicking a card used to leave the detail panel below the fold."""
    page.evaluate("window.showView('explore')")
    page.wait_for_timeout(150)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(100)
    page.evaluate("document.querySelectorAll('.wcard')[2].click()")
    page.wait_for_timeout(900)
    top = page.evaluate(
        "Math.round(document.getElementById('wdetail').getBoundingClientRect().top)"
    )
    assert top < 200, f"the detail panel sits {top}px down after selecting a window"


def test_the_view_is_reflected_in_the_url(page):
    """So a reload mid-demonstration does not lose the operator's place."""
    page.evaluate("window.showView('operate')")
    page.wait_for_timeout(150)
    assert page.evaluate("location.hash") == "#operate"

    page.goto(page.url.split("#")[0] + "#explain", wait_until="networkidle")
    page.wait_for_function("window.dashboardReady === true", timeout=30_000)
    assert page.is_visible("#v-explain"), "a deep link did not open its view"


def test_tab_semantics_and_landmarks(page):
    page.evaluate("window.showView('eval')")
    page.wait_for_timeout(150)
    assert page.evaluate("document.querySelectorAll('main').length") == 1
    assert page.get_attribute("#nav", "role") == "tablist"
    assert page.evaluate("document.querySelectorAll('nav button[role=tab]').length") == 9
    assert page.evaluate("document.querySelectorAll('[role=tabpanel]').length") == 9
    assert (
        page.evaluate("document.querySelectorAll('nav button[aria-selected=\"true\"]').length") == 1
    )
    assert page.evaluate("document.querySelectorAll('[aria-live]').length") >= 3
    assert page.get_attribute("#op-slider", "aria-label")


def test_every_interactive_control_has_a_visible_focus_style(page):
    rule = page.evaluate(
        "[...document.styleSheets].flatMap(s=>{try{return [...s.cssRules].map(r=>r.cssText)}"
        "catch(e){return []}}).find(c=>c.includes(':focus-visible')) || ''"
    )
    for selector in (
        ".wcard",
        ".btn",
        "nav button",
        ".tour-launch",
        'input[type="range"]',
        ".tourbox button",
    ):
        assert selector in rule, f"{selector} has no focus-visible style"


def test_print_and_reduced_motion_styles_exist(page):
    rules = page.evaluate(
        "[...document.styleSheets].flatMap(s=>{try{return [...s.cssRules].map(r=>r.cssText)}"
        "catch(e){return []}})"
    )
    assert any(r.startswith("@media print") for r in rules), "no print stylesheet"
    assert any("prefers-reduced-motion" in r for r in rules), "motion preference ignored"


def test_gauge_circumference_is_derived_not_hard_coded(page):
    page.evaluate("window.showView('result')")
    page.wait_for_timeout(150)
    dasharray = float(page.get_attribute("#gauge-arc", "stroke-dasharray"))
    radius = float(page.get_attribute("#gauge-arc", "r"))
    assert abs(dasharray - 2 * 3.141592653589793 * radius) < 0.01


def test_head_metadata_is_present(page):
    assert page.evaluate("!!document.querySelector('meta[name=description]')")
    assert page.evaluate("!!document.querySelector('meta[name=\"theme-color\"]')")
    assert page.get_attribute("link[rel='icon']", "href") == "favicon.svg"
    assert (paths.DASHBOARD / "favicon.svg").is_file()


def test_navigation_fits_one_row_on_a_laptop(page):
    """Nine tabs plus the title and tour launcher used to wrap to two rows."""
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_timeout(200)
    rows = page.evaluate(
        "new Set([...document.querySelectorAll('nav button')]"
        ".filter(b=>b.offsetParent!==null)"
        ".map(b=>Math.round(b.getBoundingClientRect().top))).size"
    )
    assert rows == 1, f"the navigation wraps to {rows} rows at 1440px"

    # Report mode must keep the original 1200px layout and button metrics, or
    # the report's Figures 5.1-5.6 would no longer match the page.
    page.evaluate("window.setReportMode(true)")
    page.wait_for_timeout(150)
    assert (
        page.evaluate("Math.round(document.querySelector('.app').getBoundingClientRect().width)")
        == 1200
    )
    assert (
        page.evaluate("getComputedStyle(document.querySelector('nav button')).padding")
        == "8px 14px"
    )
    page.evaluate("window.setReportMode(false)")
    page.set_viewport_size({"width": 1440, "height": 960})


def test_disclaimer_contrast_meets_aa(page):
    """The prototype caveat was the least readable text on the page at 4.27:1."""
    page.evaluate("window.showView('risk')")
    page.wait_for_timeout(150)
    ratio = page.evaluate(
        """(() => {
             const el = document.querySelector('.alertbox .msg p.caveat');
             const parse = c => c.match(/\\d+/g).slice(0,3).map(Number);
             const lum = ([r,g,b]) => {
               const v = [r,g,b].map(x => x/255)
                 .map(x => x <= 0.03928 ? x/12.92 : Math.pow((x+0.055)/1.055, 2.4));
               return 0.2126*v[0] + 0.7152*v[1] + 0.0722*v[2];
             };
             const fg = lum(parse(getComputedStyle(el).color));
             const bg = lum([28, 18, 4]);
             return (Math.max(fg,bg)+0.05) / (Math.min(fg,bg)+0.05);
           })()"""
    )
    assert ratio >= 4.5, f"disclaimer contrast is {ratio:.2f}:1, below the 4.5:1 minimum"
