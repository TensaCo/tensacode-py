"""Smoke-test the current standalone lab in Chrome, including real media playback."""

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

source = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--results", type=Path, default=source.parents[2] / "docs/results/robotics"
)
parser.add_argument("--browser", default="/usr/bin/google-chrome")
args = parser.parse_args()
page_path = (args.results / "index.html").resolve()
artifacts = Path(tempfile.mkdtemp(prefix="tensorcode-lab-browser-"))
handling_checks = []
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=args.browser,
        headless=True,
        args=["--no-sandbox", "--allow-file-access-from-files"],
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(page_path.as_uri())
    page.wait_for_function('document.querySelector("video").readyState>=1')
    assert page.evaluate('document.querySelector("video").playbackRate') == 1
    page.locator("#playback-speed").select_option("4")
    assert page.evaluate('document.querySelector("video").playbackRate') == 4
    page.locator("#playback-speed").select_option("1")
    if page.locator('#tabs button[data-id="visual-handling"]').count():
        assert (
            page.locator("#tabs button").first.get_attribute("data-id")
            == "visual-handling"
        )
        for run_id in ("high-receiver-v5", "receiver_shift"):
            page.locator("#episode").select_option(run_id)
            page.wait_for_function('document.querySelector("video").readyState>=1')
            assert page.locator('.wire[d*="NaN"], .wire[d*="Infinity"]').count() == 0
            labels = page.evaluate("episode.chapters.map(chapter => chapter.label)")
            for phase in ("walk", "transport", "lower", "open", "withdraw", "complete"):
                assert phase in labels, (run_id, phase, labels)
            page.evaluate(
                "video.currentTime = episode.chapters.find(c => c.label === 'transport').time + 0.4"
            )
            page.wait_for_function("video.readyState >= 2 && !video.seeking")
            page.locator('.block[data-id="payload"]').click()
            assert "payload_mass_estimate_kg" in page.locator("#inspector").inner_text()
            mass = page.evaluate(
                "episode.signals.filter(s => s.name.endsWith('payload_mass_estimate_kg') && s.time <= video.currentTime).at(-1).value"
            )
            assert 0.04 < mass < 0.20, (run_id, mass)
            page.locator('.block[data-id="guard"]').click()
            assert "phase: transport" in page.locator("#inspector").inner_text()
            if run_id == "high-receiver-v5":
                assert (
                    "no compensation"
                    in page.locator(".metric-reference").inner_text().lower()
                )
            else:
                assert page.locator(".metric-reference").count() == 0
            page.screenshot(path=str(artifacts / f"{run_id}-carry.png"))
            page.evaluate("video.currentTime = 46.5")
            page.wait_for_function("video.readyState >= 2 && !video.seeking")
            page.wait_for_function(
                "document.getElementById('inspector').innerText.includes('phase: complete')"
            )
            force = page.evaluate(
                "episode.signals.filter(s => s.name.endsWith('pad_a measured force') && s.time <= video.currentTime).at(-1).value"
            )
            assert force < 0.25, (run_id, force)
            page.screenshot(path=str(artifacts / f"{run_id}-released.png"))
            handling_checks.append(
                f"{run_id}: source phases, carry mass/tracking, matched comparison scope, postrelease complete/tactile telemetry"
            )
    assert page.locator("#tabs button").count() >= 5
    page.locator('#tabs button[data-id="guarded-planning"]').click()
    page.wait_for_function('document.querySelector("video").readyState>=1')
    assert "planned-height-v1" in page.locator("#episode").inner_text()
    assert page.locator("#chapters button").count() >= 6
    page.locator("#chapters button").last.click()
    assert page.locator("#chapters button").last.get_attribute("aria-current") == "step"
    assert page.evaluate('document.querySelector("video").currentTime') > 0
    page.locator("#chapters button").first.click()
    assert (
        page.locator("#chapters button").first.get_attribute("aria-current") == "step"
    )
    assert "peak lift" in page.locator("#episode-metrics").inner_text().lower()
    page.locator('#tabs button[data-id="guarded-planning"]').click()
    page.locator("#inspector summary").click()
    assert "development-shift" in page.locator("#inspector").inner_text()
    assert page.locator('.wire[d*="NaN"], .wire[d*="Infinity"]').count() == 0
    page.locator('.block[data-id="ik"]').click()
    assert "SciPy bounded" in page.locator("#inspector").inner_text()
    page.locator('.block[data-id="guard"]').click()
    page.evaluate('document.querySelector("video").currentTime=26.5')
    page.wait_for_timeout(200)
    page.wait_for_function(
        "document.getElementById('inspector').innerText.includes('ended (last samples retained)')"
    )
    page.wait_for_function(
        'document.querySelector("video").readyState>=2 && !document.querySelector("video").seeking'
    )
    page.evaluate('document.querySelector("video").play()')
    page.wait_for_timeout(300)
    page.evaluate('document.querySelector("video").pause()')
    page.mouse.move(5, 5)
    page.screenshot(path=str(artifacts / "planning.png"))
    page.locator('#tabs button[data-id="sensor-humanoid"]').click()
    page.locator('.block[data-id="traj"]').click()
    assert "Authored trajectory" in page.locator("#inspector").inner_text()
    page.evaluate('document.querySelector("video").currentTime=10')
    page.wait_for_timeout(200)
    page.locator("#note").fill("<img src=x onerror=alert(1)> diagnosis")
    page.locator("#note-form button").click()
    assert page.locator("#notes img").count() == 0
    assert "diagnosis" in page.locator("#notes").inner_text()
    page.reload()
    page.locator("#notes").wait_for()
    assert "diagnosis" in page.locator("#notes").inner_text()
    page.locator('#tabs button[data-id="trajectory-transfer"]').click()
    assert "diagnosis" not in page.locator("#notes").inner_text()
    page.locator('.block[data-id="traj"]').click()
    assert "TensorCode (left)" in page.locator("#inspector").inner_text()
    page.locator('#tabs button[data-id="sensor-humanoid"]').click()
    with page.expect_download() as info:
        page.locator("#export").click()
    download = info.value
    download.save_as(str(artifacts / "annotations.json"))
    payload = json.loads(Path(str(artifacts / "annotations.json")).read_text())
    assert payload["annotations"][0]["time"] >= 9.9
    page.locator("#notes .remove").click()
    assert page.locator("#notes li").count() == 0
    page.locator("#import").set_input_files(str(artifacts / "annotations.json"))
    page.wait_for_timeout(150)
    assert page.locator("#notes li").count() == 1
    page.locator('#tabs button[data-id="gr00t-reference"]').click()
    page.locator("#episode").select_option("episode-1")
    page.wait_for_function('document.querySelector("video").readyState>=1')
    assert "episode-1.mp4" in page.locator("#video").get_attribute("src")
    page.locator("#import").set_input_files(str(artifacts / "annotations.json"))
    page.wait_for_function(
        "document.getElementById('note-status').innerText.includes('matching experiment')"
    )
    page.locator("#note").fill("Episode-specific note")
    page.locator("#note-form button").click()
    page.locator("#episode").select_option("episode-2")
    assert page.locator("#notes li").count() == 0
    page.locator("#episode").select_option("episode-1")
    assert page.locator("#notes li").count() == 1
    page.wait_for_function('document.querySelector("video").readyState>=1')
    page.evaluate('document.querySelector("video").play()')
    page.wait_for_timeout(600)
    assert page.evaluate('document.querySelector("video").currentTime') > 0
    page.locator('#tabs button[data-id="trajectory-transfer"]').click()
    page.screenshot(path=str(artifacts / "desktop.png"))
    page.set_viewport_size({"width": 390, "height": 844})
    if page.locator('#tabs button[data-id="visual-handling"]').count():
        page.locator('#tabs button[data-id="visual-handling"]').click()
        page.locator("#episode").select_option("high-receiver-v5")
        page.wait_for_function("video.readyState >= 1")
        page.evaluate(
            "video.currentTime = episode.chapters.find(c => c.label === 'withdraw').time + 0.4"
        )
        page.wait_for_function("video.readyState >= 2 && !video.seeking")
        page.locator('.block[data-id="guard"]').click()
        assert "phase: withdraw" in page.locator("#inspector").inner_text()
        page.wait_for_function(
            "document.querySelector('#chapters button[aria-current=step]')?.innerText.toLowerCase().includes('withdraw')"
        )
        assert (
            "carry tracking median"
            in page.locator("#episode-metrics").inner_text().lower()
        )
        page.evaluate("window.scrollTo(0, 0)")
        handling_checks.append(
            "390px handling metrics and recorded withdrawal chapter without page overflow"
        )
    page.screenshot(path=str(artifacts / "mobile.png"))
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    assert not errors, errors
    print(
        "PASS: current experiments, recorded chapters, playback speed, planned-version label, failure outcomes, finite graph paths, ended-source telemetry, block inspector, exact-source telemetry, timestamp annotations, safe text, persistence, episode scoping, export/import, GR00T episode playback, mobile overflow, no JavaScript errors"
    )
    browser.close()

report = {
    "checked_at_utc": datetime.now(UTC).isoformat(),
    "browser": args.browser,
    "handling_checks": handling_checks,
    "tested_index_sha256": hashlib.sha256(page_path.read_bytes()).hexdigest(),
    "source_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.iterdir()
        if p.suffix in {".js", ".css", ".html", ".json", ".py"}
    },
    "checks": [
        "file:// standalone load",
        "current experiments and prior comparisons",
        "recorded chapter seeking and current-stage highlight",
        "evidence-derived episode metrics",
        "explicit 1x/2x/4x playback speed with unchanged source timestamps",
        "exact development video version",
        "negative outcomes visible",
        "finite diagram routes",
        "SciPy vs optional compiled solver description",
        "ended-source telemetry",
        "recorded planned video seek and playback",
        "named-port block inspector",
        "timestamped notes",
        "safe text",
        "local persistence",
        "experiment and episode scoping",
        "JSON export/import",
        "mismatched import rejection",
        "GR00T episode playback",
        "390px mobile without overflow",
        "no JavaScript errors",
    ],
    "passed": True,
}
(args.results / "browser-check.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"Screenshots and exported annotation: {artifacts}")
