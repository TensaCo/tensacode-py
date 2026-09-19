"""Real CDP transport against a disposable local Chromium, never a user's browser."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request

import pytest

from examples.general_agent.browser_connection import BrowserPlugin
from tensorcode.agent.plugin import Call


@pytest.fixture
def browser_endpoint(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        executable = driver.chromium.executable_path
    if not Path(executable).exists():
        pytest.skip("Playwright Chromium is not installed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([executable, "--headless", "--no-sandbox", "--disable-gpu",
                                f"--remote-debugging-port={port}",
                                f"--user-data-dir={tmp_path / 'profile'}", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    endpoint = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(endpoint + "/json/version", timeout=.2):
                    break
            except OSError:
                if process.poll() is not None:
                    pytest.fail("Disposable Chromium exited before CDP startup")
                time.sleep(.05)
        else:
            pytest.fail("Disposable Chromium CDP did not become available")
        yield endpoint
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture
def page_url(tmp_path):
    (tmp_path / "index.html").write_text('''<title>Transport fixture</title>
<input id="name"><button id="go" onclick="document.querySelector('#result').textContent=document.querySelector('#name').value">Submit</button><p id="result"></p>''')
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_cdp_actions_and_source_capture(browser_endpoint, page_url):
    from examples.general_agent.plugins import mount
    mounted = mount("browser:" + browser_endpoint)
    plugin = mounted.plugin
    try:
        assert mounted.descriptor()["kind"] == "browser"
        for name, args in [("navigate", {"url": page_url}),
                           ("fill", {"selector": "#name", "text": "unchanged meaning"}),
                           ("click", {"selector": "#go"})]:
            receipt = plugin.execute(Call(plugin.name, name, tuple(args.items())), key=None)
            assert receipt.status == "applied", receipt.error
        assert plugin.page.locator("#result").inner_text() == "unchanged meaning"
        observation = plugin.observe()
        assert observation["title"] == "Transport fixture"
        assert observation["screenshot"].startswith(b"\x89PNG")
        assert "unchanged meaning" in observation["html"]
        assert tuple(plugin.perceive()) == ()  # no DOM-to-belief shortcut
        assert not any(cap.effects for cap in plugin.capabilities())
        rejected = plugin.execute(Call(plugin.name, "navigate", (("url", "javascript:alert(1)"),)))
        assert rejected.status == "rejected"
    finally:
        mounted.close()
    # Disconnect does not close the explicitly connected external browser.
    with urllib.request.urlopen(browser_endpoint + "/json/version", timeout=1) as response:
        assert response.status == 200


def test_ambiguous_tabs_require_explicit_selection(browser_endpoint):
    plugin = BrowserPlugin(browser_endpoint)
    try:
        plugin.page.context.new_page()
        with pytest.raises(ValueError, match="select explicitly"):
            BrowserPlugin(browser_endpoint)
        second = BrowserPlugin(browser_endpoint + "#page=1")
        second.close()
    finally:
        plugin.close()


@pytest.mark.parametrize("endpoint", ["", "local", "file:///tmp/x", "http://localhost#page=-1"])
def test_invalid_endpoint_rejected_before_connect(endpoint):
    with pytest.raises(ValueError):
        BrowserPlugin(endpoint)
