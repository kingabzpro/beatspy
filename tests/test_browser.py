"""Optional headless Chromium checks. Install the browser extra and Chromium first."""

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

playwright = pytest.importorskip(
    "playwright.sync_api", reason="install the browser extra for frontend integration checks"
)


@pytest.fixture
def site(tmp_path):
    from beatspy.bench.demo import generate_demo_runs
    from beatspy.reporting.dashboard import render_dashboard

    results = tmp_path / "results"
    runs = generate_demo_runs(out_root=results)
    meta = json.loads((runs[0] / "run.json").read_text())
    meta["model"]["model"] = '<img src=x onerror="window.compromised=true">'
    (runs[0] / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    path = render_dashboard(results)
    from beatspy.reporting.dashboard import static_dir

    headers = json.loads((static_dir() / "vercel.json").read_text())["headers"][0]["headers"]

    class StaticHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            for header in headers:
                self.send_header(header["key"], header["value"])
            super().end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(StaticHandler, directory=str(path.parent)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", path.parent
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("width", [1280, 390])
def test_dashboard_interactions_and_safe_rendering(site, width):
    url, _ = site
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.get_by_role("button", name="demo-beta", exact=True).wait_for()
        page.get_by_role("button", name="demo-beta", exact=True).focus()
        page.keyboard.press("Enter")
        page.locator("#run-details:not([hidden])").wait_for()
        assert "Data cutoff" in page.locator("#run-meta").inner_text()
        assert page.locator("#performance svg").count() == 1
        toggle = page.locator("#performance").get_by_role("button", name="● portfolio", exact=True)
        toggle.click()
        assert toggle.get_attribute("aria-pressed") == "false"
        toggle.click()
        page.locator("#performance svg").hover()
        assert "portfolio:" in page.locator("#performance .tooltip").inner_text()
        page.get_by_role("button", name="Return", exact=False).first.click()
        assert page.locator("th[aria-sort=descending]").count() == 1
        page.locator("#trust").select_option("maintainer")
        assert "No results match" in page.locator("#status").inner_text()
        page.locator("#trust").select_option("synthetic")
        assert page.locator("#leaderboard-table tbody tr").count() == 3
        attack = page.get_by_role("button", name='<img src=x onerror="window.compromised=true">', exact=True)
        attack.click()
        page.locator("#run-details:not([hidden])").wait_for()
        assert page.locator("#run-title img").count() == 0
        page.reload()
        page.locator("#run-details:not([hidden])").wait_for()
        assert page.locator("#decisions details").count() > 0
        page.locator("#decisions summary").first.click()
        assert page.locator("#decisions details").first.get_attribute("open") is not None
        page.keyboard.press("Tab")
        assert not errors
        # The horizontal table may scroll inside its wrapper; the page must fit mobile.
        assert page.locator("body").bounding_box()["width"] <= width
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        browser.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_all_years_average_and_yearly_drilldown(site, width):
    url, directory = site
    catalog = json.loads((directory / "data/index.json").read_text())
    catalog["runs"] = catalog["runs"][:2]
    catalog.pop("selected", None)
    for row, year, value in zip(catalog["runs"], ["2025", "2026"], [0.1, 0.3], strict=True):
        row.update(model="demo-beta", year=year, start=f"{year}-10-03", end=f"{year}-12-31", scenario=f"{year}-recent")
        row["metrics"].update(total_return=value, spy_total_return=0.02, excess_return_vs_spy=value - 0.02)
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/data/index.json", lambda route: route.fulfill(json=catalog))
        page.goto(url)
        page.get_by_role("button", name="demo-beta", exact=True).click()
        page.locator("#run-details:not([hidden])").wait_for()
        assert page.locator("#scenario").input_value() == "2026-recent"
        page.locator("#year").select_option("")
        assert page.locator("#scenario").input_value() == ""
        assert page.locator("#leaderboard-table tbody tr").count() == 1
        assert "20.00%" in page.locator("#leaderboard-table tbody").inner_text()
        assert "2025, 2026" in page.locator("#leaderboard-table tbody").inner_text()
        page.get_by_role("button", name="Avg Return", exact=False).click()
        page.get_by_role("button", name="demo-beta", exact=True).focus()
        page.keyboard.press("Enter")
        page.locator("#average-details:not([hidden])").wait_for()
        assert not page.locator("#run-details").is_visible()
        assert page.locator("#average-runs tbody tr").count() == 2
        assert page.evaluate("location.hash") == "#model=demo-beta"
        page.reload()
        page.locator("#average-details:not([hidden])").wait_for()
        assert page.locator("#year").input_value() == ""
        assert "20.00%" in page.locator("#leaderboard-table tbody").inner_text()
        page.locator("#average-runs").get_by_role("link", name="2025", exact=True).click()
        page.locator("#run-details:not([hidden])").wait_for()
        assert page.locator("#year").input_value() == "2025"
        assert page.evaluate("location.hash").startswith("#run=")
        assert not page.locator("#average-details").is_visible()
        assert "10.00%" in page.locator("#leaderboard-table tbody").inner_text()
        page.locator("#year").select_option("")
        page.get_by_role("button", name="demo-beta", exact=True).click()
        page.locator("#average-details:not([hidden])").wait_for()
        page.locator("#trust").select_option("maintainer")
        assert "No results match" in page.locator("#status").inner_text()
        assert not page.locator("#average-details").is_visible()
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        assert not errors
        browser.close()


def test_empty_and_failed_fetch_states(site):
    url, directory = site
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.route("**/data/index.json", lambda route: route.fulfill(json={"schema_version": 1, "runs": []}))
        page.goto(url)
        page.get_by_text("No published runs yet.", exact=False).wait_for()
        page.unroute("**/data/index.json")
        page.route("**/data/index.json", lambda route: route.fulfill(status=500, body="unavailable"))
        page.reload()
        page.get_by_text("Results unavailable:", exact=False).wait_for()
        page.unroute("**/data/index.json")
        page.goto(url)
        page.get_by_role("button", name="demo-beta", exact=True).wait_for()
        page.route("**/equity_curve.csv", lambda route: route.fulfill(status=404, body="missing"))
        page.get_by_role("button", name="demo-beta", exact=True).click()
        page.get_by_text("Unable to load this run:", exact=False).wait_for()
        assert not page.locator("#run-details").is_visible()
        page.unroute("**/equity_curve.csv")
        catalog = json.loads((directory / "data/index.json").read_text())
        catalog["runs"][0]["dir"] = "../secret"
        page.route("**/data/index.json", lambda route: route.fulfill(json=catalog))
        page.reload()
        page.get_by_role("button", name=catalog["runs"][0]["model"], exact=True).click()
        page.get_by_text("Invalid artifact path", exact=False).wait_for()
        assert not page.locator("#run-details").is_visible()
        browser.close()
