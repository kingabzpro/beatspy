"""Optional headless Chromium checks. Install the browser extra and Chromium first."""

import json
import struct
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
        assert page.locator("#model-tokens .bar-row").count() == 3
        assert "Estimated model cost —" in page.locator("#cost-summary").inner_text()
        assert page.locator("#running-cost svg").count() == 0
        page.get_by_text("Pricing assumptions & calculator", exact=True).click()
        page.locator("#input-price").fill("2")
        page.locator("#output-price").fill("8")
        assert page.locator("#running-cost svg").count() == 1
        assert page.locator("#agent-costs .bar-row").count() == 5
        assert "Custom rates" in page.locator("#price-note").inner_text()
        page.get_by_role("button", name="Est. cost", exact=False).click()
        page.get_by_role("button", name="Est. cost", exact=False).click()
        assert page.locator("#leaderboard-table tbody tr").first.get_by_role("button").inner_text() == "demo-beta"
        page.locator("#input-price").fill("0")
        page.locator("#output-price").fill("0")
        assert "Estimated model cost $0.0000" in page.locator("#cost-summary").inner_text()
        page.get_by_role("button", name="Reset rates", exact=True).click()
        assert "Estimated model cost —" in page.locator("#cost-summary").inner_text()
        toggle = page.locator("#performance").get_by_role("button", name="● portfolio", exact=True)
        toggle.click()
        assert toggle.get_attribute("aria-pressed") == "false"
        toggle.click()
        page.locator("#performance svg").hover()
        assert "portfolio:" in page.locator("#performance .tooltip").inner_text()
        page.get_by_role("button", name="Return", exact=False).first.click()
        assert page.locator("th[aria-sort=descending]").count() == 1
        assert page.locator("#trust, #year, #scenario").count() == 0
        assert page.locator("#leaderboard-table tbody tr").count() == 3
        attack = page.get_by_role("button", name='<img src=x onerror="window.compromised=true">', exact=True)
        attack.click()
        page.locator("#run-details:not([hidden])").wait_for()
        assert page.locator("#run-title img").count() == 0
        page.reload()
        page.locator("#run-details:not([hidden])").wait_for()
        assert page.locator("#decisions, #trades, #decision-tokens, #cost-decisions").count() == 0
        page.keyboard.press("Tab")
        assert not errors
        # The horizontal table may scroll inside its wrapper; the page must fit mobile.
        assert page.locator("body").bounding_box()["width"] <= width
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        browser.close()


@pytest.mark.parametrize("width", [1280, 390])
def test_latest_per_model_and_brand_alignment(site, width):
    url, directory = site
    catalog = json.loads((directory / "data/index.json").read_text())
    catalog["runs"] = catalog["runs"][:2]
    catalog.pop("selected", None)
    for row, year, created, value in zip(
        catalog["runs"], ["2026", "2025"], ["2026-10-01", "2026-10-04"], [0.9, 0.1], strict=True
    ):
        row.update(model="demo-beta", year=year, start=f"{year}-10-03", end=f"{year}-12-31", created_utc=created)
        row["metrics"].update(total_return=value, excess_return_vs_spy=value)
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/data/index.json", lambda route: route.fulfill(json=catalog))
        page.goto(url)
        page.get_by_role("button", name="demo-beta", exact=True).wait_for()
        assert page.locator("#leaderboard-table tbody tr").count() == 1
        assert "10.00%" in page.locator("#leaderboard-table tbody").inner_text()
        assert "2025-10-03" in page.locator("#leaderboard-table tbody").inner_text()
        page.get_by_role("button", name="demo-beta", exact=True).click()
        page.locator("#run-details:not([hidden])").wait_for()
        assert "Unsigned" in page.locator("#verification").inner_text()
        assert "github.com/kingabzpro/beatspy/issues/new" in page.locator("#submit-result").get_attribute("href")
        logo = page.locator(".brandmark").bounding_box()
        text = page.locator(".brand span").bounding_box()
        assert abs((logo["y"] + logo["height"] / 2) - (text["y"] + text["height"] / 2)) < 1
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
        page.get_by_text("No new results yet.", exact=False).wait_for()
        assert page.locator("#run-count").inner_text() == "0 results"
        assert page.locator("#leaderboard-table tbody tr").count() == 0
        assert page.locator("#run-details").is_hidden()
        assert page.locator("#cost-overview").is_hidden()
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


def test_brand_assets_metadata_and_keyboard_navigation(site):
    url, _ = site
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        assert page.locator('link[rel="canonical"]').get_attribute("href") == "https://beatspy.vercel.app/"
        assert page.locator('meta[property="og:type"]').get_attribute("content") == "website"
        assert page.locator('meta[name="twitter:card"]').get_attribute("content") == "summary_large_image"
        assert page.locator('meta[property="og:image:alt"]').get_attribute("content")
        for name, dimensions in [
            ("social-preview.png", (1200, 630)),
            ("apple-touch-icon.png", (180, 180)),
            ("icon-192.png", (192, 192)),
            ("icon-512.png", (512, 512)),
        ]:
            response = page.request.get(f"{url}/assets/{name}")
            assert response.ok
            content = response.body()
            assert content[:8] == b"\x89PNG\r\n\x1a\n"
            assert struct.unpack(">II", content[16:24]) == dimensions
        for asset in ("favicon.ico", "assets/favicon.svg", "site.webmanifest", "robots.txt", "sitemap.xml"):
            assert page.request.get(f"{url}/{asset}").ok
        manifest = page.request.get(f"{url}/site.webmanifest").json()
        assert manifest["short_name"] == "BeatSPY"
        for icon in manifest["icons"]:
            assert page.request.get(f"{url}/{icon['src']}").ok
        assert page.locator("#decisions, #trades, #decision-tokens, #cost-decisions").count() == 0
        page.keyboard.press("Tab")
        assert page.locator(".skip-link").evaluate("element => element === document.activeElement")
        page.keyboard.press("Enter")
        assert page.evaluate("document.activeElement.id") == "main-content"
        browser.close()
