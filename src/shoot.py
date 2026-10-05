import sys, time
from playwright.sync_api import sync_playwright

views = ["overview", "predict", "result", "risk", "explain", "eval"]
outnames = {"overview": "fig5_1_overview", "predict": "fig5_2_forecast_config",
            "result": "fig5_3_prediction_result", "risk": "fig5_4_risk_alert",
            "explain": "fig5_5_explainability", "eval": "fig5_6_model_evaluation"}

with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium/chrome-linux/chrome" if False else None)
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    page.goto("http://localhost:8791/index.html")
    page.wait_for_timeout(500)
    for v in views:
        page.evaluate(f"window.showView('{v}')")
        page.wait_for_timeout(250)
        page.screenshot(path=f"figures/{outnames[v]}.png")
        print("saved", outnames[v])
    browser.close()
