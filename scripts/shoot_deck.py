"""Capture tight, purpose-built crops of the dashboard for the review deck.

    python -m solarflare dashboard --port 8820     # in one terminal
    python scripts/shoot_deck.py [port] [out_dir]  # in another

These are deliberately not the same captures as `solarflare screenshots`: those
are whole views for the documentation, these are single cards cropped for
slides, taken at 2x and with the replay driven to a frame where the region is
actually flaring. scripts/make_deck.py embeds them.
"""

import pathlib
import sys

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8820"
OUT = pathlib.Path(
    sys.argv[2]
    if len(sys.argv) > 2
    else pathlib.Path(__file__).resolve().parent.parent / "docs/deck_assets"
)
OUT.mkdir(parents=True, exist_ok=True)
URL = f"http://localhost:{PORT}/index.html"

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(
        viewport={"width": 1440, "height": 1000}, device_scale_factor=2, reduced_motion="reduce"
    )
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL, wait_until="domcontentloaded")
    pg.wait_for_function("window.dashboardReady === true", timeout=30_000)
    pg.wait_for_timeout(600)

    # --- replay, driven to a frame where the region is actually flaring ---
    pg.evaluate("window.showView('replay')")
    pg.wait_for_timeout(300)
    inputs = pg.evaluate(
        "[...document.querySelectorAll('#v-replay input')]"
        ".map(e=>({type:e.type,id:e.id,max:e.max,value:e.value}))"
    )
    print("replay inputs:", inputs)
    sel = "#v-replay input[type=range]"
    if pg.query_selector(sel):
        hi = pg.evaluate(
            "(s)=>{s.value=Math.round(s.max*0.78);"
            "s.dispatchEvent(new Event('input',{bubbles:true}));"
            "return s.value+'/'+s.max;}",
            pg.query_selector(sel),
        )
        print("replay frame set to", hi)
    pg.wait_for_timeout(600)
    print("  dial now:", pg.inner_text("#rp-pct"), "|", pg.inner_text("#rp-band"))
    pg.locator("#v-replay .card").filter(has=pg.locator("#rp-play")).first.screenshot(
        path=str(OUT / "d_replay.png")
    )

    # --- evaluation: the KPI strip and the confusion matrix ---
    pg.evaluate("window.showView('eval')")
    pg.wait_for_timeout(300)
    pg.locator("#eval-kpi").screenshot(path=str(OUT / "d_eval_kpi.png"))
    pg.locator("#v-eval .card").filter(has=pg.locator("#cm-grid")).first.screenshot(
        path=str(OUT / "d_confusion.png")
    )

    # --- the risk banding a user would see ---
    pg.evaluate("window.showView('risk')")
    pg.wait_for_timeout(300)
    pg.locator("#v-risk .card").filter(has=pg.locator("#riskrow")).first.screenshot(
        path=str(OUT / "d_risk.png")
    )

    # --- the operating-point trade-off ---
    pg.evaluate("window.showView('operate')")
    pg.wait_for_timeout(300)
    pg.locator("#v-operate .card").filter(has=pg.locator("#op-slider")).first.screenshot(
        path=str(OUT / "d_operate.png")
    )

    print("page errors:", errs if errs else "none")
    b.close()

for f in sorted(OUT.glob("d_*.png")):
    print(f"  {f.name:<18} {f.stat().st_size//1024:>5} KB")
