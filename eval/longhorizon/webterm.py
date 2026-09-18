"""A real shell in a sandbox, shown as a browser page an agent can perceive and type into.

    python -m eval.longhorizon.webterm --workdir /path/to/task --port 8801 [--tools /path/to/venv]

bash runs on a pty inside bubblewrap: no network, read-only system directories, the task
directory mounted at /app, a throwaway home at /home/agent, and an optional read-only
tool environment at /opt/tools. The page mirrors the accessible structure of a desktop
terminal (a "Terminal" region with prompt lines, output blocks and a "Shell input" box),
so agents that read terminals by screen work unchanged.

The server also keeps a full, agent-invisible log (GET /log) for evaluation.
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import os
import pty
import re
import select
import signal
import socketserver
import subprocess
import threading
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][A-Za-z0-9]|\r")
MARK = re.compile(r"\x1e(\d+)\x1f([^\x1e]*)\x1e")
USER, HOST = "agent", "sandbox"


def bwrap_command(workdir: Path, tools: Path | None, network: bool = False) -> list[str]:
    cmd = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--hostname", HOST]
    if not network:
        cmd.append("--unshare-net")
    for d in ("/usr", "/bin", "/sbin", "/lib", "/etc"):
        if Path(d).exists():
            cmd += ["--ro-bind", d, d]
    if Path("/lib64").exists():
        cmd += ["--ro-bind", "/lib64", "/lib64"]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/home", "--dir", "/home/agent", "--bind", str(workdir), "/app"]
    path = "/usr/local/bin:/usr/bin:/bin"
    if tools:
        cmd += ["--ro-bind", str(tools), "/opt/tools"]
        path = "/opt/tools/bin:" + path
    cmd += ["--chdir", "/app", "--clearenv", "--setenv", "HOME", "/home/agent", "--setenv", "USER", USER, "--setenv", "PATH", path, "--setenv", "TERM", "dumb",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "MPLBACKEND", "Agg", "--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
    return cmd


class Shell:
    """One interactive bash; commands are delimited by a prompt hook that reports exit status and cwd."""

    def __init__(self, argv: list[str]) -> None:
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execvp(argv[0], argv + ["bash", "--noprofile", "--norc", "-i"])
        self.lock = threading.Lock()
        self.blocks: list[dict] = []  # {cmd, out, exit, cwd, started, ended}
        self.cwd = "/app"
        self.cleared_at = 0
        self.version = 0
        self._buf = ""
        self._ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()
        os.write(self.fd, b"stty -echo; PS1=''; PS2=''; PROMPT_COMMAND='printf \"\\036%d\\037%s\\036\" \"$?\" \"$PWD\"'\n")
        self._ready.wait(10)

    def _read(self) -> None:
        while True:
            try:
                r, _, _ = select.select([self.fd], [], [], 0.5)
                if not r:
                    continue
                data = os.read(self.fd, 65536).decode("utf-8", "replace")
            except OSError:
                return
            with self.lock:
                self._buf += ANSI.sub("", data)
                while (m := MARK.search(self._buf)) is not None:
                    before, self._buf = self._buf[: m.start()], self._buf[m.end():]
                    if self.blocks and self.blocks[-1]["exit"] is None:
                        b = self.blocks[-1]
                        b["out"] += before
                        b["exit"], b["cwd"], b["ended"] = int(m[1]), m[2], time.time()
                    self.cwd = m[2]
                    self._ready.set()
                if self.blocks and self.blocks[-1]["exit"] is None and self._buf:
                    self.blocks[-1]["out"] += self._buf
                    self._buf = ""
                self.version += 1

    def run(self, command: str) -> None:
        with self.lock:
            if command.strip() in ("clear", "reset", "tput clear") and not (self.blocks and self.blocks[-1]["exit"] is None):
                self.cleared_at = len(self.blocks)  # a screen operation: earlier output leaves the screen (still in /log)
                self.version += 1
                return
            if self.blocks and self.blocks[-1]["exit"] is None:
                # still running: forward as input to the running program
                self.blocks[-1]["out"] += command + "\n"
            else:
                self.blocks.append({"cmd": command, "out": "", "exit": None, "cwd": self.cwd, "started": time.time(), "ended": None})
            self.version += 1
        os.write(self.fd, (command + "\n").encode())

    def interrupt(self) -> None:
        os.write(self.fd, b"\x03")

    def close(self) -> None:
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


PAGE = """<!doctype html><title>Terminal</title><meta charset="utf-8">
<style>
 body{margin:0;background:#1e1f24;color:#e6e6e6;font:14px/1.45 ui-monospace,Menlo,monospace;height:100vh;display:grid;grid-template-rows:auto 1fr auto}
 header{padding:6px 12px;background:#2b2d33;display:flex;gap:16px;align-items:baseline} header div{color:#9aa}
 #screen{overflow-y:auto;padding:8px 12px} .p{color:#8fd694;white-space:pre-wrap} .o{white-space:pre-wrap;margin:0} .x{color:#f28b82}
 form{display:flex;gap:8px;padding:6px 12px;background:#26282e} form span{color:#8fd694;white-space:nowrap} input{flex:1;background:#111;color:#eee;border:1px solid #444;font:inherit;padding:4px}
</style>
<section aria-label="Terminal" style="display:contents">
<header><div class="t">Terminal</div><div id="meta">bash · sandbox</div><button type="button" id="intr" aria-label="Interrupt">Ctrl-C</button></header>
<div id="screen"></div>
<form id="f"><span id="ps"></span><input id="in" aria-label="Shell input" autocomplete="off" spellcheck="false"></form>
</section>
<script>
let version=-1;
const esc=(s)=>s.replace(/[&<>]/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function poll(){
  try{ const r=await fetch('state?since='+version); const s=await r.json();
    if(s.version!==version){ version=s.version; render(s); } }catch(e){}
  setTimeout(poll,120);
}
function render(s){
  const scr=document.getElementById('screen'); let h='';
  for(const b of s.blocks){
    h+='<div class="p">'+esc(s.user+'@'+s.host+':'+b.cwd+'$ '+b.cmd)+'</div>';
    for(const chunk of b.chunks) h+='<div class="o">'+esc(chunk)+'</div>';
    if(b.exit===null) h+='<div class="o">…</div>'; else if(b.exit!==0) h+='<div class="o x">[exit '+b.exit+']</div>';
  }
  const running=s.blocks.length&&s.blocks[s.blocks.length-1].exit===null;
  if(!running) h+='<div class="p">'+esc(s.user+'@'+s.host+':'+s.cwd+'$')+'</div>';
  scr.innerHTML=h; scr.scrollTop=scr.scrollHeight;
  document.getElementById('meta').textContent='bash · sandbox · '+s.total+' history entries';
}
document.getElementById('f').onsubmit=async(e)=>{e.preventDefault();const i=document.getElementById('in');const v=i.value;i.value='';
  await fetch('run',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({command:v})});};
document.getElementById('intr').onclick=()=>fetch('interrupt',{method:'POST'});
poll();
</script>"""


def chunks(text: str, lines: int = 30, chars: int = 1800) -> list[str]:
    out, cur, n = [], [], 0
    for line in text.rstrip("\n").split("\n") if text.strip() else []:
        while len(line) > chars:
            out.append(line[:chars])
            line = line[chars:]
        if len(cur) >= lines or n + len(line) > chars:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def serve(shell: Shell, port: int, keep_blocks: int = 40) -> http.server.HTTPServer:
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def _json(self, obj: object, code: int = 200) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                data = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(data)
            elif path == "/state":
                with shell.lock:
                    blocks = shell.blocks[shell.cleared_at:][-keep_blocks:]
                    view = [{"cmd": b["cmd"], "cwd": b["cwd"], "exit": b["exit"], "chunks": chunks(b["out"][-60000:])} for b in blocks]
                    self._json({"version": shell.version, "blocks": view, "cwd": shell.cwd, "user": USER, "host": HOST, "total": len(shell.blocks)})
            elif path == "/log":
                with shell.lock:
                    self._json(shell.blocks)
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path == "/run":
                shell.run(str(body.get("command", "")))
                self._json({"ok": True})
            elif self.path == "/interrupt":
                shell.interrupt()
                self._json({"ok": True})
            else:
                self.send_error(404)

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = S(("127.0.0.1", port), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def start(workdir: Path, port: int, tools: Path | None) -> tuple[Shell, http.server.HTTPServer]:
    shell = Shell(bwrap_command(workdir, tools))
    return shell, serve(shell, port)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--tools", type=Path, default=None)
    args = ap.parse_args()
    shell, server = start(args.workdir.resolve(), args.port, args.tools)
    print(f"terminal: http://127.0.0.1:{args.port}/", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        shell.close()


if __name__ == "__main__":
    main()
