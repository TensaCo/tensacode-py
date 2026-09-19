"""Durable local chat UI and CLI server."""
from __future__ import annotations

import argparse
import base64
import collections
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from examples.browser_agents import harness

PAGE = Path(__file__).parent / "chat.html"


class Hub:
    """Fan events out to viewers: every other event in order, and the newest frame per source.

    Frames are dropped rather than queued — a viewer that falls behind should see the current
    picture, not a backlog of old ones — and there is one current picture *per plugin*, since
    any number of machines may be mounted.
    """

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.clients: list[dict] = []
        self.history: collections.deque = collections.deque(maxlen=600)
        self.frames: dict[str, dict] = {}

    def publish(self, ev: dict) -> None:
        with self.lock:
            if ev["type"] == "frame":
                source = (ev.get("chat_id"), ev.get("source", ""))
                self.frames[source] = ev
                for c in self.clients:
                    if c["chat_id"] is None or c["chat_id"] == ev.get("chat_id"):
                        c["frames"][source] = ev
            else:
                self.history.append(ev)
                for c in self.clients:
                    if c["chat_id"] is None or c["chat_id"] == ev.get("chat_id"):
                        c["events"].append(ev)
            self.lock.notify_all()

    def serve(self, handler, chat_id=None) -> None:
        with self.lock:
            client = {"chat_id": chat_id, "events": collections.deque(e for e in self.history if chat_id is None or e.get("chat_id") == chat_id), "frames": {k: e for k, e in self.frames.items() if chat_id is None or e.get("chat_id") == chat_id}}
            self.clients.append(client)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        try:
            while True:
                with self.lock:
                    while not client["events"] and not client["frames"]:
                        if not self.lock.wait(timeout=15):
                            break
                    batch = list(client["events"])
                    client["events"].clear()
                    frames, client["frames"] = list(client["frames"].values()), {}
                out = "".join(f"data: {json.dumps(ev)}\n\n" for ev in batch + frames)
                handler.wfile.write((out or ": keepalive\n\n").encode())
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.clients.remove(client)


def worker(inbox, events, fps, reader=None, specs=()):
    from examples.general_agent.plugins import attach_agent, mount, specs_descriptors
    from examples.general_agent.connections import ConnectionRegistry
    from tensorcode.agent import Agent
    from dataclasses import replace
    sessions = {}
    configured = specs_descriptors(specs)
    specification = {d["id"]: spec for d, spec in zip(configured, specs)}
    while True:
        job = inbox.get()
        if job is None:
            for agent, registry, grammar in sessions.values():
                for descriptor in registry.descriptors():
                    registry.get(descriptor['id']).close()
            return
        chat_id, message_id = job['chat_id'], job['message_id']
        def emit(event):
            events.put({**event, 'chat_id': chat_id, 'message_id': message_id, 't': time.time()})
        emit({'type': 'busy', 'busy': True})
        error, started = False, time.perf_counter()
        try:
            if chat_id not in sessions:
                agent = Agent([], reader=reader)
                sessions[chat_id] = (agent, ConnectionRegistry(), agent.grammar)
            agent, registry, base_grammar = sessions[chat_id]
            known = {d['id'] for d in registry.descriptors()}
            for ident in job['connection_ids']:
                if ident not in known:
                    adapter = mount(specification[ident])
                    adapter.id = ident
                    existing_names = {d['name'] for d in registry.descriptors()}
                    base_name, suffix = adapter.name, 2
                    while adapter.name in existing_names:
                        adapter.name = f'{base_name}#{suffix}'
                        suffix += 1
                    adapter.plugin.name = adapter.name
                    registry.register(adapter)
            actual = {d['id']: d for d in registry.descriptors()}
            emit({'type': 'connections', 'connections': [actual.get(d['id'], d) for d in configured]})
            selected = registry.select(job['connection_ids'])
            agent.plugins = [m.plugin for m in selected]
            entries = [entry for plugin in agent.plugins for entry in plugin.lexicon]
            agent.grammar = replace(base_grammar, lexicon=base_grammar.lexicon.extend(*entries)) if entries else base_grammar
            attach_agent(selected, agent)
            images = []
            for attachment in job['attachments']:
                media = attachment['media_type']
                source = agent.interpretations.add_source(attachment['name'],
                    modality='image' if media.startswith('image/') else 'video' if media.startswith('video/') else 'file',
                    provider='chat-attachment', payload=attachment['data'],
                    metadata={k: v for k, v in attachment.items() if k != 'data'})
                emit({'type': 'attachment_evidence', 'attachment_id': attachment['id'], 'source_id': source.id, 'understood': False})
                if media.startswith('image/'):
                    images.append(attachment['data'])
            turn = agent.turn(job['text'], images=images)
            for event in turn.events:
                emit(event)
            emit({'type': 'assistant_result', 'text': turn.reply or 'No response was produced for this input.', 'seconds': turn.seconds})
            for connection in selected:
                try:
                    picture = connection.view()
                    if picture:
                        emit({'type': 'frame', 'source': connection.name, 'connection_id': connection.id, 'data': base64.b64encode(picture).decode()})
                except Exception as exc:
                    emit({'type': 'note', 'text': f'Preview unavailable: {type(exc).__name__}'})
        except Exception as exc:
            error = True
            traceback.print_exc()
            emit({'type': 'assistant_result', 'text': f'The turn could not complete: {type(exc).__name__}: {exc}', 'error': True})
        finally:
            emit({'type': 'busy', 'busy': False, 'error': error, 'seconds': round(time.perf_counter() - started, 2)})


def main():
    from examples.general_agent.chat_store import ChatStore
    from examples.general_agent.chat_api import ChatApplication
    from examples.general_agent.plugins import specs_descriptors
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--no-open', action='store_true')
    ap.add_argument('--reader', default='learned', choices=['learned', 'grammar'])
    ap.add_argument('--plugin', action='append', default=None)
    ap.add_argument('--data-dir', type=Path, default=Path.home() / '.cache' / 'tensorcode' / 'chat')
    ap.add_argument('--import-history', type=Path)
    args = ap.parse_args()
    specs = tuple(s for s in (args.plugin or ['desktop', 'self']) if s and s != 'none')
    ctx = mp.get_context('spawn')
    inbox, events = ctx.Queue(), ctx.Queue()
    store, hub = ChatStore(args.data_dir), Hub()
    if args.import_history:
        store.import_history(json.loads(args.import_history.read_text()))
    app = ChatApplication(store, inbox, hub, specs_descriptors(specs))
    base, server = harness.serve(app.routes, port=args.port)
    proc = ctx.Process(target=worker, args=(inbox, events, args.fps, None if args.reader == 'grammar' else args.reader, specs), daemon=True)
    proc.start()
    print(f'general agent: {base}/ (history: {args.data_dir})', flush=True)
    if not args.no_open:
        webbrowser.open(f'{base}/')
    try:
        while proc.is_alive():
            try:
                app.receive(events.get(timeout=1))
            except queue.Empty:
                continue
    finally:
        server.shutdown()
        inbox.put(None)
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        store.close()


if __name__ == '__main__':
    main()
