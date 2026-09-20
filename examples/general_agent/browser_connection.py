"""Explicit CDP connection to an existing Chromium tab.

This is transport and action execution, not learned browser understanding. Pixels
and DOM remain raw observations. No textual request is interpreted in this adapter.
"""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit, parse_qs
from threading import local

from tensorcode.agent.plugin import Call, Capability, Param, Plugin
from tensorcode.outcomes import Receipt


_drivers = local()


def _acquire_driver():
    # Playwright permits one synchronous event loop per worker thread. Multiple
    # chat connections share transport ownership without sharing selected tabs.
    if not getattr(_drivers, "users", 0):
        from playwright.sync_api import sync_playwright
        _drivers.driver = sync_playwright().start()
        _drivers.users = 0
    _drivers.users += 1
    return _drivers.driver


def _release_driver():
    _drivers.users -= 1
    if not _drivers.users:
        _drivers.driver.stop()
        del _drivers.driver


class BrowserPlugin(Plugin):
    def __init__(self, endpoint: str, *, timeout_ms: int = 5000) -> None:
        super().__init__(name="browser:" + endpoint)
        parts = urlsplit(endpoint)
        if parts.scheme not in {"http", "https", "ws", "wss"} or not parts.netloc:
            raise ValueError("browser requires an explicit CDP http(s) or ws(s) endpoint")
        options = parse_qs(parts.fragment, strict_parsing=True) if parts.fragment else {}
        if set(options) - {"page"} or ("page" in options and len(options["page"]) != 1):
            raise ValueError("browser fragment accepts only #page=N")
        index = int(options["page"][0]) if "page" in options else None
        if index is not None and index < 0:
            raise ValueError("browser page index must be nonnegative")
        self._driver = _acquire_driver()
        try:
            self.browser = self._driver.chromium.connect_over_cdp(
                urlunsplit(parts._replace(fragment="")), timeout=timeout_ms)
            pages = [page for context in self.browser.contexts for page in context.pages]
            if index is None and len(pages) != 1:
                raise ValueError(f"CDP exposes {len(pages)} tabs; select explicitly with #page=N")
            chosen = 0 if index is None else index
            if chosen >= len(pages):
                raise ValueError(f"browser page {chosen} does not exist")
            self.page = pages[chosen]
            self.page.set_default_timeout(timeout_ms)
            self.page.set_default_navigation_timeout(timeout_ms)
        except Exception:
            _release_driver()
            raise
        self.closed = False

    def close(self) -> None:
        """Detach the driver; never close the user's browser or tab."""
        if not self.closed:
            _release_driver()
            self.closed = True

    def capabilities(self):
        return (
            Capability("navigate", (Param("url", "url"),), effect_kind="external",
                       description="Navigate the explicitly connected browser tab."),
            Capability("click", (Param("selector", "selector"),), effect_kind="external",
                       description="Click one uniquely matching explicit Playwright selector."),
            Capability("fill", (Param("selector", "selector"), Param("text", "text")),
                       effect_kind="external", description="Fill one uniquely matching editable element."),
        )

    def screenshot(self) -> bytes:
        return self.page.screenshot(type="png")

    def observe(self) -> dict:
        """Retain raw source evidence; DOM labels are not established scene semantics."""
        return {"url": self.page.url, "title": self.page.title(), "html": self.page.content(),
                "screenshot": self.screenshot(), "media_type": "image/png",
                "provenance": "browser-cdp", "limitations": ["Uninterpreted DOM and pixels"]}

    def observe_evidence(self) -> dict:
        """Expose the same raw observation to the cognitive evidence boundary."""
        return self.observe()

    def execute(self, act: Call, *, key: str | None = None) -> Receipt:
        if self.closed:
            return Receipt(act, "rejected", error="Browser connection is closed")
        args = dict(act.args)
        if len(args) != len(act.args):
            return Receipt(act, "rejected", error="Browser argument names must be unique")
        capability = next((cap for cap in self.capabilities() if cap.name == act.capability), None)
        if act.plugin != self.name or capability is None:
            return Receipt(act, "rejected", error="Unknown browser capability/provider")
        if set(args) != {param.name for param in capability.params} or not all(
                isinstance(value, str) for value in args.values()):
            return Receipt(act, "rejected", error="Browser arguments must match declared string parameters")
        if act.capability == "navigate" and urlsplit(args["url"]).scheme not in {"http", "https"}:
            return Receipt(act, "rejected", error="Navigation requires an explicit http(s) URL")
        try:
            if act.capability == "navigate":
                self.page.goto(args["url"], wait_until="domcontentloaded")
            elif act.capability == "click":
                self.page.locator(args["selector"]).click()
            else:
                self.page.locator(args["selector"]).fill(args["text"])
        except Exception as exc:
            return Receipt(act, "indeterminate", error=f"{type(exc).__name__}: {exc}")
        return Receipt(act, "applied", idempotency_key=key)
