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
    assert page.locator("#events").count() == 0
    page.locator('#filters select[data-field="policy"]').select_option("GR00T N1.6")
    assert page.locator("#tabs button:visible").count() == 1
    assert (
        page.locator("#tabs button:visible").get_attribute("data-id")
        == "gr00t-reference"
    )
    page.locator('#filters select[data-field="policy"]').select_option("")
    assert page.evaluate("""() => {
        const tree = evidenceTree('Scene', {robots: [{id: 'g1-a'}], instruction: '<img src=x onerror=alert(1)>'});
        return tree.querySelectorAll('img').length === 0
            && tree.textContent.includes('<img src=x onerror=alert(1)>')
            && tree.querySelectorAll('details').length >= 2;
    }""")
    page.evaluate(
        "window.savedPreview = current.language_preview; current.language_preview = {command:'python -m example_server'}; renderLanguagePreview()"
    )
    assert page.locator("#language-form button").is_disabled()
    assert "python -m example_server" in page.locator("#language-status").inner_text()
    page.evaluate(
        "current.language_preview = window.savedPreview; renderLanguagePreview()"
    )
    assert page.evaluate("""() => {
        const target = document.createElement('div');
        renderPolicyComparison(target, {path:'fixture.json', complete_and_matched:false, scope:'Test fixture only', rows:[
            {policy:'<img src=x onerror=alert(1)>',case:'c0',status:'physical failure',seconds:3.2,termination:'fall',robots:{r0:{strict_success:false}}},
            {policy:'pending',case:'c0',status:'unrun'}
        ]});
        const selector = target.querySelector('select');
        selector.value = '1'; selector.onchange();
        return target.textContent.includes('comparison not admitted') && selector.options.length===2 && !target.querySelector('img') && target.querySelectorAll('tr').length===3;
    }""")
    page.locator("#playback-speed").select_option("4")
    assert page.evaluate('document.querySelector("video").playbackRate') == 4
    page.locator("#playback-speed").select_option("1")
    if page.locator('#tabs button[data-id="warehouse-six-jobs"]').count():
        assert (
            page.locator("#tabs button").first.get_attribute("data-id")
            == "warehouse-six-jobs"
        )
        assert page.evaluate("current.metadata.fleet") == 3
        assert page.locator("#episode-metrics strong").first.inner_text() == "6"
        assert page.locator("#chapters select option").count() == 3
        assert "4B LANGUAGE" in page.locator("#graph-title").inner_text()
        assert page.evaluate("current.graph.blocks.parser.kind") == "f"
        page.wait_for_function("video.readyState >= 2")
        assert page.evaluate("video.duration") > 125
        page.evaluate("video.currentTime = 90")
        page.wait_for_function("!video.seeking && video.readyState >= 2")
        assert "B ·" in page.locator("#chapters button[aria-current=step]").inner_text()
        page.screenshot(path=str(artifacts / "six-jobs-main.png"))
        for robot in ["0", "1", "2"]:
            page.locator("#chapters select").select_option(robot)
            page.locator("#chapters button").last.click()
            page.wait_for_function("!video.seeking && video.readyState >= 2")
            assert (
                "complete"
                in page.locator("#chapters button[aria-current=step]")
                .inner_text()
                .lower()
            )
        assert "offline audit" in page.locator("#inspection-trees").inner_text().lower()
    if page.locator('#tabs button[data-id="warehouse-fleet-alg2"]').count():
        page.locator('#tabs button[data-id="warehouse-fleet-alg2"]').click()
        assert page.evaluate("current.metadata.fleet") == 3
        if page.locator('#filters select[data-field="study"]').count():
            page.locator('#filters select[data-field="study"]').select_option(
                "Offline perception replay"
            )
            assert page.locator("#tabs button:visible").count() == 1
            page.locator('#filters select[data-field="study"]').select_option("")
            assert page.locator(".visual-evidence[open]").count() == 0
            page.locator(".visual-evidence > summary").first.click()
            page.wait_for_function(
                "document.querySelector('.visual-evidence img').naturalWidth > 0"
            )
            assert (
                "not a controller intervention"
                in page.locator(".visual-evidence").first.inner_text()
            )
            page.locator(".visual-evidence img").first.scroll_into_view_if_needed()
            page.screenshot(path=str(artifacts / "offline-perception.png"))
            page.locator(".visual-evidence > summary").first.click()
            page.locator('#filters select[data-field="study"]').select_option(
                "Semantic vision diagnostic"
            )
            assert page.locator("#tabs button:visible").count() == 1
            page.locator('#filters select[data-field="study"]').select_option("")
            semantic = page.locator(".visual-evidence").filter(
                has=page.locator(
                    "summary", has_text="Semantic vision · offline failures retained"
                )
            )
            semantic.locator("> summary").click()
            assert (
                semantic.locator("a", has_text="Open diagnostic").get_attribute("href")
                == "warehouse/semantic-vision-study/index.html"
            )
            assert (
                "4/12" in semantic.inner_text()
                and "2 false elephant" in semantic.inner_text()
            )
            assert "Not a robot policy" in semantic.inner_text()
            page.screenshot(path=str(artifacts / "supporting-diagnostics.png"))
            semantic.locator("> summary").click()
            assert (
                "historical failed-trial"
                in page.locator("#inspection-trees").inner_text()
            )
            assert (
                "not fresh inference"
                in page.locator("#inspection-trees").text_content().lower()
            )
            page.evaluate("document.querySelector('main').scrollTo(0,0)")

        assert page.locator("#chapters select option").count() == 3
        assert "ALG.2" in page.locator("#graph-title").inner_text()
        page.locator("#chapters select").select_option("2")
        assert page.evaluate("episode.chapter_label") == "r2"
        page.locator("#chapters button").last.click()
        page.wait_for_function("video.readyState >= 2 && !video.seeking")
        assert (
            "complete"
            in page.locator("#chapters button[aria-current=step]").inner_text().lower()
        )
        page.evaluate("video.play()")
        page.wait_for_timeout(300)
        page.evaluate("video.pause()")
        page.screenshot(path=str(artifacts / "three-parallel-alg2-v3.png"))
    if page.locator('#tabs button[data-id="warehouse-fleet"]').count():
        assert page.locator("#tabs button").first.get_attribute("data-id") in {
            "warehouse-fleet",
            "warehouse-fleet-alg2",
            "warehouse-six-jobs",
        }
        page.locator('#tabs button[data-id="warehouse-fleet"]').click()
        assert page.locator("#language-form button").is_disabled()
        assert (
            "initial-scene-graph.json" in page.locator("#language-status").inner_text()
        )
        for episode_id in page.locator("#episode option").evaluate_all(
            "options => options.map(o => o.value)"
        ):
            page.locator("#episode").select_option(episode_id)
            page.wait_for_function("video.readyState >= 1")
            if episode_id == "model-parallel-v3":
                assert (
                    page.locator("#presentation-still").get_attribute("href")
                    == "warehouse/parallel-cycles.png"
                )
                assert "not the policy camera" in page.locator(
                    "#presentation-still"
                ).get_attribute("title")
            else:
                assert page.locator("#presentation-still").is_hidden()
            assert page.locator("#chapters select option").count() == 2
            page.locator("#chapters select").select_option("1")
            assert page.evaluate("episode.chapter_label") == "r1"
            page.locator("#chapters button").last.click()
            page.wait_for_function("video.readyState >= 2 && !video.seeking")
            assert (
                page.locator("#chapters button").last.get_attribute("aria-current")
                == "step"
            )
            assert page.locator('.wire[d*="NaN"], .wire[d*="Infinity"]').count() == 0
            assert page.locator("#inspection-trees details[open]").count() == 0
            page.evaluate("video.play()")
            page.wait_for_timeout(300)
            page.evaluate("video.pause()")
            page.screenshot(path=str(artifacts / f"{episode_id}.png"))
    if page.locator('#tabs button[data-id="tactile-recovery"]').count():
        page.locator('#tabs button[data-id="tactile-recovery"]').click()
        page.wait_for_function("video.readyState>=1")
        assert abs(page.evaluate("video.duration") - 85) < 0.1
        assert page.locator("#chapters select option").count() == 4
        assert "Recovery r0" in page.evaluate("episode.chapter_label")
        assert page.locator("#episode-metrics strong").first.inner_text() == "2"
        assert (
            "1 baseline / 2" in page.locator(".metric-reference").inner_text().lower()
        )
        page.evaluate(
            "video.currentTime=episode.chapters.find(c=>c.label==='reacquire retract').time+0.4"
        )
        page.wait_for_function("video.readyState>=2 && !video.seeking")
        page.locator('.block[data-id="retry"]').click()
        assert "recovery_attempts: 1" in page.locator("#inspector").inner_text()
        page.evaluate("video.play()")
        page.wait_for_timeout(300)
        page.evaluate("video.pause()")
        page.screenshot(path=str(artifacts / "tactile-retry.png"))
        page.locator("#chapters select").select_option("2")
        page.evaluate("video.currentTime=84")
        page.wait_for_function("video.readyState>=2 && !video.seeking")
        assert (
            "failed"
            in page.locator("#chapters button[aria-current=step]").inner_text().lower()
        )
        assert (
            page.evaluate(
                "episode.signals.filter(s=>s.name==='Recovery r0 · right · phase' && s.time<=84).at(-1).value"
            )
            == "complete"
        )
        page.screenshot(path=str(artifacts / "tactile-pair-final.png"))
        page.locator("#episode").select_option("recovery-cycles-20-40")
        page.wait_for_function("video.readyState>=1")
        assert abs(page.evaluate("video.duration") - 20) < 0.1
        assert "source 20–40 s" in page.locator("#episode option:checked").inner_text()
        assert page.evaluate("episode.signals.every(s=>s.time>=0 && s.time<20)")
        retry_time = page.evaluate(
            "episode.chapters.find(c=>c.label==='reacquire retract').time"
        )
        assert abs(retry_time - 1.84) < 0.02
        page.evaluate("video.currentTime=2.3")
        page.wait_for_function("video.readyState>=2 && !video.seeking")
        if (
            page.locator('.block[data-id="retry"]').get_attribute("aria-pressed")
            != "true"
        ):
            page.locator('.block[data-id="retry"]').click()
        assert "recovery_attempts: 1" in page.locator("#inspector").inner_text()
        page.screenshot(path=str(artifacts / "recovery-cycles.png"))
    if page.locator('#tabs button[data-id="visual-handling"]').count():
        page.locator('#tabs button[data-id="visual-handling"]').click()
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
    page.locator("#inspector-panel > summary").click()
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
    if page.locator('#tabs button[data-id="psi01-reference"]').count():
        page.locator('#tabs button[data-id="psi01-reference"]').click()
        assert page.locator("#episode option").count() == 5
        for option in page.locator("#episode option").evaluate_all(
            "options => options.map(o => o.value)"
        ):
            page.locator("#episode").select_option(option)
            page.wait_for_function("video.readyState >= 1")
            assert page.evaluate(
                "Number.isFinite(video.duration) && video.duration > 0"
            )
        assert "18" in page.locator("#episode-metrics").inner_text()
        assert "19" in page.locator(".metric-reference").inner_text()
        page.evaluate("video.play()")
        page.wait_for_timeout(400)
        page.evaluate("video.pause()")
        assert page.evaluate("video.currentTime > 0")
        page.screenshot(path=str(artifacts / "psi01-reference.png"))
    if page.locator('#tabs button[data-id="psi0-reference"]').count():
        page.locator('#tabs button[data-id="psi0-reference"]').click()
        assert page.locator("#episode option").count() == 2
        for option in page.locator("#episode option").evaluate_all(
            "options => options.map(o => o.value)"
        ):
            page.locator("#episode").select_option(option)
            page.wait_for_function("video.readyState >= 1")
            assert (
                "0.5× sim speed" in page.locator("#episode option:checked").inner_text()
            )
            page.evaluate("video.play()")
            page.wait_for_timeout(300)
            page.evaluate("video.pause()")
            assert page.evaluate("video.currentTime > 0")
        assert page.locator("#inspection-trees details[open]").count() == 0
        page.screenshot(path=str(artifacts / "psi0-reference.png"))
    if page.locator('#tabs button[data-id="pi0-reference"]').count():
        page.locator('#tabs button[data-id="pi0-reference"]').click()
        assert page.evaluate("current.metadata.fleet") == 0
        assert page.locator("#episode option").count() == 2
        assert page.locator("#episode-metrics strong").first.inner_text() == "0"
        for episode_id in page.locator("#episode option").evaluate_all(
            "options => options.map(o => o.value)"
        ):
            page.locator("#episode").select_option(episode_id)
            page.wait_for_function("video.readyState >= 1")
            assert (
                "failed transfer"
                in page.locator("#episode option:checked").inner_text()
            )
            page.evaluate("video.currentTime = 5.5")
            page.wait_for_function("video.readyState >= 2 && !video.seeking")
            page.evaluate("video.play()")
            page.wait_for_timeout(300)
            page.evaluate("video.pause()")
            assert page.evaluate("video.currentTime > 5.5")
            page.screenshot(path=str(artifacts / f"{episode_id}.png"))
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
    page.evaluate("video.play()")
    page.wait_for_timeout(300)
    page.evaluate("video.pause()")
    page.mouse.move(5, 5)
    page.screenshot(path=str(artifacts / "mobile.png"))
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    if page.locator('#tabs button[data-id="warehouse-fleet-alg2"]').count():
        page.locator('#tabs button[data-id="warehouse-fleet-alg2"]').click()
        page.wait_for_function("video.readyState >= 1")
        page.evaluate("video.currentTime=40")
        page.wait_for_function("video.readyState>=2 && !video.seeking")
        page.evaluate("video.play()")
        page.wait_for_timeout(300)
        page.evaluate("video.pause();window.scrollTo(0,0)")
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(artifacts / "three-mobile.png"))
    if page.locator('#tabs button[data-id="warehouse-six-jobs"]').count():
        page.locator('#tabs button[data-id="warehouse-six-jobs"]').click()
        page.wait_for_function("video.readyState >= 1")
        page.evaluate("video.currentTime=90")
        page.wait_for_function("video.readyState>=2 && !video.seeking")
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(artifacts / "six-jobs-mobile.png"))
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
        "metadata filters and no action-log panel",
        "recorded chapter seeking and current-stage highlight",
        "evidence-derived episode metrics",
        "nested scene/task evidence uses safe text",
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
