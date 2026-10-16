"""Check the actual local language service through the warehouse dashboard."""

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8765/")
parser.add_argument(
    "--results",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "docs/results/robotics",
)
args = parser.parse_args()
artifacts = Path(tempfile.mkdtemp(prefix="tensorcode-preview-browser-"))
responses = []
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"]
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.url)
    page.locator('#tabs button[data-id="warehouse-fleet"]').click()
    assert page.locator("#language-form button").is_enabled()
    for mode, text, expected in [
        (
            "instruction",
            "move parcel_0 to receiver_0 while move parcel_1 to receiver_1",
            "supported",
        ),
        ("question", "How many parcels?", "supported"),
        ("question", "Where is parcel_9?", "unknown"),
    ]:
        page.locator("#language-mode").select_option(mode)
        page.locator("#language-text").fill(text)
        with page.expect_response(
            lambda response: response.url.endswith("/api/warehouse/interpret")
        ) as received:
            page.locator("#language-form button").click()
        response = received.value
        payload = response.json()
        assert response.ok
        assert payload["scope"] == "recorded_observation_preview"
        assert payload["execution"] is False
        assert payload["motion_feasibility"] == "not_checked"
        assert payload["status"] == expected, payload
        page.wait_for_function(
            "document.querySelector('#language-status').innerText.includes('no execution')"
        )
        assert page.locator("#language-result details[open]").count() == 0
        if mode == "instruction":
            assert len(payload["tasks"]) == 2
        elif expected == "supported":
            assert payload["answer"]["details"]["observed_count"] == 2
            assert "2 parcels" in page.locator("#language-result").inner_text()
        responses.append({"mode": mode, "text": text, "response": payload})
    page.screenshot(path=str(artifacts / "preview.png"), full_page=True)
    assert not errors, errors
    browser.close()
report = {
    "checked_at_utc": datetime.now(UTC).isoformat(),
    "url": args.url,
    "passed": True,
    "tested_index_sha256": hashlib.sha256(
        (args.results / "index.html").read_bytes()
    ).hexdigest(),
    "responses": responses,
    "screenshots": str(artifacts),
    "scope": "Actual local API; recorded scene preview, no robot execution or motion feasibility claim",
}
(args.results / "preview-browser-check.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(f"PASS actual instruction / QA / abstention preview: {artifacts}")
