"""Drive the live chat assistant as a black box, and read the simulator's own state.

The assistant is a server (``--port 8770``) over a Seed computer. We never import its
code here: a probe sends a message, reads the replies off its event stream, and checks
the world with the simulator's API. That keeps the grader independent of the agent even
though both are ours.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

ASSISTANT = "http://127.0.0.1:8770"
SEED = "http://127.0.0.1:4391"
COMPUTER = "ubuntu-2"


@dataclass
class Turn:
    """One exchange: what we said, how it was parsed, what came back."""

    said: str
    replies: list[str] = field(default_factory=list)
    acts: list[str] = field(default_factory=list)
    seconds: float = 0.0
    model_calls: int = 0
    status: str = ""

    @property
    def reply(self) -> str:
        return "\n".join(self.replies)


class Chat:
    """A conversation with the running assistant, over its HTTP/SSE interface."""

    def __init__(self, base: str = ASSISTANT, timeout: float = 120.0) -> None:
        self.base, self.timeout = base, timeout
        self.events: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        time.sleep(1.5)  # let the stream attach before the first message

    def _listen(self) -> None:
        try:
            host = self.base.split("//", 1)[1]
            name, _, port = host.partition(":")
            sock = socket.create_connection((name, int(port or 80)), timeout=10)
            sock.sendall(b"GET /events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n")
            sock.settimeout(5)
            buf = b""
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.startswith(b"data: "):
                        try:
                            self.events.append(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            return

    def close(self) -> None:
        self._stop.set()

    def say(self, text: str) -> Turn:
        mark = len(self.events)
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(f"{self.base}/say", data=body, method="POST", headers={"content-type": "application/json"})
        urllib.request.urlopen(req, timeout=15).read()
        deadline = time.time() + self.timeout
        turn = Turn(said=text)
        while time.time() < deadline:
            time.sleep(0.4)
            fresh = self.events[mark:]
            done = [e for e in fresh if e.get("type") == "busy" and e.get("busy") is False]
            if done:
                for e in fresh:
                    if e.get("type") == "chat" and e.get("from") == "agent":
                        turn.replies.append(e.get("text", ""))
                    elif e.get("type") == "understood":
                        turn.acts = [f.get("act", "?") for f in e.get("frames", [])]
                turn.seconds = float(done[-1].get("seconds") or 0.0)
                turn.model_calls = int(done[-1].get("model_calls") or 0)
                turn.status = str(done[-1].get("status") or "")
                return turn
        turn.status = "timeout"
        return turn


def shell(command: str, computer: str = COMPUTER) -> dict:
    """Read or change the simulated machine directly: the probe's independent ground truth."""
    body = json.dumps({"command": command}).encode()
    req = urllib.request.Request(f"{SEED}/api/computers/{computer}/shell", data=body, method="POST", headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


def listing(path: str, computer: str = COMPUTER) -> list[str]:
    out = shell(f"ls -1pA {path}", computer)
    return [ln for ln in out.get("stdout", "").splitlines() if ln.strip()]


def read_file(path: str, computer: str = COMPUTER) -> str | None:
    try:
        url = f"{SEED}/api/computers/{computer}/file?path={urllib.parse.quote(path)}"
        return json.loads(urllib.request.urlopen(url, timeout=15).read()).get("content")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None


def up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3).read(1)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False
