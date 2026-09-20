"""Explicit CDP connection to an existing Chromium tab.

This is transport and action execution, not learned browser understanding. Pixels
and DOM remain raw observations. No textual request is interpreted in this adapter.
"""
from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit, parse_qs
from threading import local
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from tensorcode.agent.plugin import Call, Capability, Param, Plugin
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.learning.experience import _same


@dataclass(frozen=True)
class BrowserDocumentCapture:
    """Detached raw capture; only its issuing live adapter can authenticate it."""

    id: str
    snapshot: dict


@dataclass(frozen=True)
class BrowserDocumentTargetEvidence:
    """Transport identity only, never a predictive feature or activation permit."""

    token: str
    capture_id: str
    connection_id: str
    session_id: str
    frame_id: str
    frame_loaders: tuple[tuple[str, str], ...]
    backend_node_id: int
    document_index: int
    node_index: int
    action: Call


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
        self._document_session = None
        self._document_page = self.page
        self._document_captures = {}
        self._document_targets = {}
        self._document_target_evidence = {}
        self._document_observations = {}
        self._document_connection_id = uuid4().hex
        self._document_session_id = uuid4().hex

    def close(self) -> None:
        """Detach the driver; never close the user's browser or tab."""
        if not self.closed:
            if self._document_session is not None:
                try:
                    self._document_session.detach()
                except Exception:
                    pass  # Closing an already disconnected transport is harmless.
            self._document_captures.clear()
            self._document_targets.clear()
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
            Capability("activate_node", (Param("target", "browser_document_target"),),
                       effect_kind="external",
                       description="Programmatically activate one authenticated HTML node; not a physical click."),
        )

    def screenshot(self) -> bytes:
        # Playwright's default caret hiding writes inline styles to controls and
        # can leave style="" behind. Evidence capture must not edit the document
        # whose exact identity an action guard is about to validate.
        return self.page.screenshot(type="png", caret="initial")

    def document_snapshot(self) -> dict:
        """Read literal CDP document structure without interpreting screenshot pixels.

        This capture is independent of screenshots. It does not certify visual
        visibility, affordances, application concepts, or cross-capture identity.
        """
        if self.closed:
            raise RuntimeError("browser connection is closed")
        return self._cdp().send("DOMSnapshot.captureSnapshot", {"computedStyles": []})

    def _cdp(self):
        if self.closed:
            raise RuntimeError("browser connection is closed")
        if self.page is not self._document_page:
            raise RuntimeError("browser connection page identity changed")
        if self._document_session is None:
            self._document_session = self.page.context.new_cdp_session(self.page)
        return self._document_session

    def _document_state(self):
        session = self._cdp()
        # Frame loader identities distinguish navigation even if the new HTML is
        # byte-for-byte identical. The session is bound to this exact page.
        before = session.send("Page.getFrameTree")
        snapshot = self.document_snapshot()
        after = session.send("Page.getFrameTree")
        if not _same(before, after):
            raise RuntimeError("document changed while capturing")
        return snapshot, after

    @staticmethod
    def _frame_loaders(identity):
        def descend(tree):
            frame = tree["frame"]
            yield frame["id"], frame["loaderId"]
            for child in tree.get("childFrames", ()):
                yield from descend(child)
        return tuple(sorted(descend(identity["frameTree"])))

    def capture_document(self) -> BrowserDocumentCapture:
        snapshot, identity = self._document_state()
        capture_id = uuid4().hex
        self._document_captures[capture_id] = (deepcopy(snapshot), deepcopy(identity))
        return BrowserDocumentCapture(capture_id, deepcopy(snapshot))

    def authenticate_document_capture(self, capture):
        if self.closed:
            return Unknown("browser_closed")
        if type(capture) is not BrowserDocumentCapture or type(capture.id) is not str:
            return Unknown("foreign_document_capture")
        retained = self._document_captures.get(capture.id)
        if retained is None:
            return Unknown("foreign_document_capture")
        if not _same(capture.snapshot, retained[0]):
            return Unknown("altered_document_capture")
        return True

    def prepare_document_target(self, capture, document_index, node_index):
        valid = self.authenticate_document_capture(capture)
        if valid is not True:
            return valid
        if type(document_index) is not int or type(node_index) is not int or min(document_index, node_index) < 0:
            return Unknown("invalid_document_node")
        snapshot, identity = self._document_captures[capture.id]
        try:
            document = snapshot["documents"][document_index]
            nodes = document["nodes"]
            backend_id = nodes["backendNodeId"][node_index]
            frame_id = snapshot["strings"][document["frameId"]]
            if type(backend_id) is not int or backend_id <= 0 or nodes["nodeType"][node_index] != 1:
                return Unknown("unsupported_document_node", "Activation requires an element, never an inferred parent")
        except (IndexError, KeyError, TypeError):
            return Unknown("invalid_document_node")
        token = uuid4().hex
        self._document_targets[token] = (capture.id, backend_id, frame_id)
        valid = self.validate_document_target(token)
        if valid is not True:
            self._document_targets.pop(token, None)
            return valid
        self._document_target_evidence[token] = BrowserDocumentTargetEvidence(
            token, capture.id, self._document_connection_id, self._document_session_id,
            frame_id, self._frame_loaders(identity), backend_id, document_index, node_index,
            Call(self.name, "activate_node", (("target", token),)))
        return token

    def document_target_evidence(self, token):
        """Export issued identity even after its one-use action token is consumed."""
        evidence = self._document_target_evidence.get(token) if type(token) is str else None
        return deepcopy(evidence) if evidence is not None else Unknown("unknown_document_target")

    def authenticate_document_target_evidence(self, evidence):
        if type(evidence) is not BrowserDocumentTargetEvidence or type(evidence.token) is not str:
            return Unknown("foreign_document_target_evidence")
        retained = self._document_target_evidence.get(evidence.token)
        if retained is None or not _same(evidence, retained):
            return Unknown("altered_or_foreign_document_target_evidence")
        return True

    def authenticate_document_observation(self, observation):
        if type(observation) is not dict or type(observation.get("document_observation_id")) is not str:
            return Unknown("foreign_document_observation")
        retained = self._document_observations.get(observation["document_observation_id"])
        if retained is None or not _same(observation, retained):
            return Unknown("altered_or_foreign_document_observation")
        return True

    def validate_document_target_observation(self, evidence, observation):
        """Authenticate target continuity, allowing observed state to change.

        This does not assert causality or establish before/action/after ordering.
        Consumers must authenticate the action and workspace attempt provenance.
        """
        for result in (self.authenticate_document_target_evidence(evidence),
                       self.authenticate_document_observation(observation)):
            if result is not True:
                return result
        inline = tuple(value for value in observation["document_targets"] if value.token == evidence.token)
        if len(inline) != 1 or not _same(inline[0], evidence):
            return Unknown("document_target_not_in_observation")
        identity = observation["document_identity"]
        if (identity["connection_id"] != evidence.connection_id or
                identity["session_id"] != evidence.session_id or
                identity["frame_loaders"] != evidence.frame_loaders):
            return Unknown("document_target_identity_changed")
        snapshot = observation["document_snapshot"]
        matches = []
        for document in snapshot["documents"]:
            if snapshot["strings"][document["frameId"]] == evidence.frame_id:
                matches.extend(index for index, backend_id in enumerate(document["nodes"]["backendNodeId"])
                               if backend_id == evidence.backend_node_id)
        if len(matches) != 1:
            return Unknown("document_target_missing_or_replaced")
        return True

    def validate_document_target(self, token):
        if self.closed:
            return Unknown("browser_closed")
        target = self._document_targets.get(token) if type(token) is str else None
        if target is None:
            return Unknown("unknown_or_consumed_document_target")
        snapshot, identity = self._document_captures[target[0]]
        try:
            current, current_identity = self._document_state()
        except Exception as exc:
            return Unknown("document_unavailable", str(exc))
        # Deliberately conservative: no ref rebinding across document changes,
        # including identical replacement nodes, scroll/layout, or form state.
        if not _same(identity, current_identity) or not _same(snapshot, current):
            return Unknown("stale_document_target")
        return True

    def _activate_document_target(self, act, token, key):
        valid = self.validate_document_target(token)
        target = self._document_targets.pop(token, None)  # every attempt consumes it
        if valid is not True or target is None:
            return Receipt(act, "rejected", error=valid.reason if isinstance(valid, Unknown) else "Consumed target")
        session, object_id = self._cdp(), None
        try:
            resolved = session.send("DOM.resolveNode", {"backendNodeId": target[1]})
            object_id = resolved.get("object", {}).get("objectId")
            if not object_id:
                return Receipt(act, "rejected", error="Target node is no longer resolvable")
            # Final document read precedes activation. The browser is external;
            # CDP does not offer an atomic snapshot-comparison/dispatch operation.
            snapshot, identity = self._document_captures[target[0]]
            current, current_identity = self._document_state()
            if not _same(snapshot, current) or not _same(identity, current_identity):
                return Receipt(act, "rejected", error="stale_document_target")
            result = session.send("Runtime.callFunctionOn", {
                "objectId": object_id,
                "functionDeclaration": "function() { if (!this.isConnected || !(this instanceof HTMLElement)) return false; HTMLElement.prototype.click.call(this); return true; }",
                "returnByValue": True,
            })
            if "exceptionDetails" in result:
                return Receipt(act, "indeterminate", error="Node activation raised in the browser")
            if result.get("result", {}).get("value") is not True:
                return Receipt(act, "rejected", error="Target is detached or does not support HTML activation")
        except Exception as exc:
            return Receipt(act, "indeterminate", error=f"{type(exc).__name__}: {exc}")
        finally:
            if object_id:
                try:
                    session.send("Runtime.releaseObject", {"objectId": object_id})
                except Exception:
                    pass
        return Receipt(act, "applied", idempotency_key=key)

    def observe(self) -> dict:
        """Retain raw source evidence; DOM labels are not established scene semantics."""
        return {"url": self.page.url, "title": self.page.title(), "html": self.page.content(),
                "screenshot": self.screenshot(), "media_type": "image/png",
                "provenance": "browser-cdp", "limitations": ["Uninterpreted DOM and pixels"]}

    def observe_evidence(self) -> dict:
        """Issue authentic raw evidence; identity fields carry no learned meaning."""
        observation = self.observe()
        snapshot, identity = self._document_state()
        observation = {**observation, "document_snapshot": snapshot,
            "document_observation_id": uuid4().hex,
            "document_targets": tuple(deepcopy(value) for value in self._document_target_evidence.values()),
            "document_identity": {"connection_id": self._document_connection_id,
                "session_id": self._document_session_id,
                "frame_loaders": self._frame_loaders(identity)}}
        self._document_observations[observation["document_observation_id"]] = deepcopy(observation)
        return observation

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
        if act.capability == "activate_node":
            return self._activate_document_target(act, args["target"], key)
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
