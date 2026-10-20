"""Real Chromium checks against the running localhost frontend and API.

Run separately from backend pytest: python -m scripts.browser_checks.
Only the connection-error step intercepts traffic; success/retry use real HTTP.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unittest

from playwright.sync_api import expect, sync_playwright

from scripts.smoke_test import SAMPLE_TEXT, validate_prediction

FRONTEND = "http://127.0.0.1:8080"
PREDICT = FRONTEND + "/api/predict"


class FrontendBrowserChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        cls.artifacts = Path(__file__).resolve().parents[1] / "outputs/browser" / stamp
        cls.artifacts.mkdir(parents=True, exist_ok=False)
        print(f"Browser screenshots/traces: {cls.artifacts}", flush=True)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        cls.browser = cls.playwright.chromium.launch(headless=True)
        cls.addClassCleanup(cls.browser.close)
        expect.set_options(timeout=35000)

    def setUp(self):
        # A new context isolates page state and network interception for each case.
        self.context = self.browser.new_context(service_workers="block")
        self.addCleanup(self.context.close)
        self.context.tracing.start(screenshots=True, snapshots=True, sources=True)
        self.addCleanup(self.save_evidence)
        self.page = self.context.new_page()
        self.page.set_default_timeout(35000)
        self.page.goto(FRONTEND, wait_until="load")
        expect(self.page.get_by_role("heading", name="From text to insight.")).to_be_visible()
        self.report = self.page.get_by_label("Report text", exact=True)
        self.analyze = self.page.get_by_role("button", name="Analyze text")
        self.sample = self.page.get_by_role("button", name="Use sample")
        # HTML <output> also has implicit status role; target the application's message.
        self.status = self.page.locator("#status")

    def save_evidence(self):
        try:
            if hasattr(self, "page"):
                self.page.screenshot(path=str(self.artifacts / f"{self._testMethodName}.png"), full_page=True)
        finally:
            self.context.tracing.stop(path=str(self.artifacts / f"{self._testMethodName}.zip"))

    def submit_real_prediction(self):
        with self.page.expect_response(lambda response: response.url == PREDICT
                                       and response.request.method == "POST") as received:
            self.analyze.click()
        response = received.value
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.request.post_data), {"text": SAMPLE_TEXT})
        prediction = response.json()
        validate_prediction(prediction)
        expect(self.status).to_have_text(
            "Analysis complete. Results are from synthetic demo models.")
        expect(self.page.locator("#results")).to_be_visible()
        expect(self.page.locator("#severity-label")).to_have_text(prediction["severity"]["label"])
        for selector, key in (("#ade-entities", "ADE"), ("#drug-entities", "DRUG")):
            expect(self.page.locator(selector).locator("li")).to_have_text(
                prediction["entities"][key] or ["None returned"])
        for label, probability in prediction["severity"]["probabilities"].items():
            bar = self.page.get_by_role("progressbar", name=f"{label} probability", exact=True)
            self.assertAlmostEqual(float(bar.get_attribute("value")), probability, places=6)
            expect(bar.locator("..").locator("output")).to_have_text(f"{probability * 100:.1f}%")
        expect(self.analyze).to_be_enabled()
        expect(self.sample).to_be_enabled()
        expect(self.report).to_be_editable()
        expect(self.page.locator("#results-panel")).to_have_attribute("aria-busy", "false")

    def test_sample_fills_report(self):
        self.sample.click()
        expect(self.report).to_have_value(SAMPLE_TEXT)
        expect(self.page.locator("#character-count")).to_have_text(f"{len(SAMPLE_TEXT)} / 5,000")
        expect(self.page.locator("#results")).to_be_hidden()

    def test_real_prediction_is_rendered_and_edit_clears_results(self):
        self.sample.click()
        self.submit_real_prediction()
        self.report.fill("A different fictional report.")
        expect(self.page.locator("#results")).to_be_hidden()
        expect(self.page.locator("#empty-state")).to_be_visible()

    def test_empty_and_whitespace_input_never_reach_api(self):
        requests = []
        self.page.on("request", lambda request: requests.append(request.url))
        self.analyze.click()
        self.assertFalse(self.report.evaluate("element => element.validity.valid"))
        self.report.fill("   ")
        self.analyze.click()
        self.assertEqual(self.report.evaluate("element => element.validationMessage"),
                         "Enter some text before analyzing.")
        self.assertNotIn(PREDICT, requests)
        expect(self.page.locator("#results")).to_be_hidden()

    def test_connection_error_allows_real_retry(self):
        self.sample.click()
        # Browser-local failure injection: no Docker/API shutdown or success mock.
        self.page.route(PREDICT, lambda route: route.abort("connectionrefused"))
        self.analyze.click()
        expect(self.status).to_contain_text("Cannot reach the API.")
        expect(self.status).to_have_class(re.compile(r"\berror\b"))
        expect(self.page.locator("#results")).to_be_hidden()
        expect(self.analyze).to_be_enabled()
        expect(self.sample).to_be_enabled()
        expect(self.report).to_be_editable()
        self.page.unroute(PREDICT)
        self.submit_real_prediction()
        expect(self.status).not_to_have_class(re.compile(r"\berror\b"))

    def test_gateway_error_allows_real_retry(self):
        self.sample.click()
        # Simulate Nginx's upstream-failure response, then retry through real Nginx.
        self.page.route(PREDICT, lambda route: route.fulfill(
            status=502, content_type="text/html", body="<h1>Bad Gateway</h1>"))
        self.analyze.click()
        expect(self.status).to_contain_text("Analysis failed (HTTP 502).")
        expect(self.page.locator("#results")).to_be_hidden()
        expect(self.analyze).to_be_enabled()
        expect(self.report).to_be_editable()
        self.page.unroute(PREDICT)
        self.submit_real_prediction()


if __name__ == "__main__":
    unittest.main(verbosity=2)
