import base64
import io
import json
import queue

import pytest

from examples.general_agent.chat_store import ChatStore
from examples.general_agent.chat_api import ChatApplication
from examples.general_agent.server import Hub


class Handler:
    def __init__(self, path, method='GET', body=None, headers=None):
        self.path, self.command = path, method
        raw = json.dumps(body).encode() if body is not None else b''
        self.headers = {'Content-Length': str(len(raw)), **(headers or {})}
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.response_headers = {}
    def send_response(self, status):
        self.status = status
    def send_header(self, key, value):
        self.response_headers[key] = value
    def end_headers(self):
        pass
    def json(self):
        return json.loads(self.wfile.getvalue())


def test_history_reloads_without_replaying_pending_turns(tmp_path):
    store = ChatStore(tmp_path)
    chat = store.create_chat()
    attachment = store.upload('notes.txt', 'text/plain', base64.b64encode(b'original bytes').decode())
    message = store.add_message(chat['id'], 'user', 'Read this', attachment_ids=[attachment['id']], status='queued')
    store.close()
    store = ChatStore(tmp_path)
    assert store.chat(chat['id'])['status'] == 'interrupted'
    assert store.messages(chat['id'])[0]['status'] == 'interrupted'
    assert store.messages(chat['id'])[0]['attachments'][0]['id'] == attachment['id']
    assert store.attachment(attachment['id'], content=True)['data'] == b'original bytes'
    assert store.message(message['id'])['text'] == 'Read this'
    store.close()


def test_ui_and_cli_common_ingestion_and_lifecycle(tmp_path):
    store, inbox, hub = ChatStore(tmp_path), queue.Queue(), Hub()
    app = ChatApplication(store, inbox, hub)
    chat = store.create_chat()
    ui = Handler(f"/api/chats/{chat['id']}/messages", 'POST', {'text': 'hello'})
    app.routes(ui)
    assert ui.status == 202
    job = inbox.get_nowait()
    app.receive({'type': 'busy', 'chat_id': chat['id'], 'message_id': job['message_id'], 'busy': True})
    app.receive({'type': 'assistant_result', 'chat_id': chat['id'], 'message_id': job['message_id'], 'text': 'response'})
    app.receive({'type': 'busy', 'chat_id': chat['id'], 'message_id': job['message_id'], 'busy': False})
    cli = Handler('/say', 'POST', {'text': 'scripted', 'chat_id': chat['id']})
    app.routes(cli)
    assert cli.status == 202
    assert [m['origin'] for m in store.messages(chat['id'])] == ['ui', 'agent', 'cli']
    assert len({e['message']['id'] for e in hub.history if e['type'] == 'message'}) == 3
    assert len(store.chats()) == 1
    for _ in range(2):
        app.routes(Handler('/say', 'POST', {'text': 'same default cli chat'}))
    assert len(store.chats()) == 2
    store.close()


@pytest.mark.parametrize('body', [
    {'name': 'a', 'media_type': 'text/plain', 'data': '!notbase64'},
    {'name': 'a', 'media_type': 'text/plain\r\nx: bad', 'data': ''},
    {'name': '../x', 'media_type': 'text/plain', 'data': None},
])
def test_invalid_uploads_rejected(tmp_path, body):
    store = ChatStore(tmp_path)
    app = ChatApplication(store, queue.Queue(), Hub())
    handler = Handler('/api/attachments', 'POST', body)
    app.routes(handler)
    assert handler.status == 400
    assert store.attachments() == []
    store.close()


def test_limits_and_resource_ids(tmp_path, monkeypatch):
    import examples.general_agent.chat_store as module
    monkeypatch.setattr(module, 'MAX_ATTACHMENT_BYTES', 3)
    store = ChatStore(tmp_path)
    with pytest.raises(ValueError):
        store.upload('a', 'text/plain', base64.b64encode(b'1234').decode())
    with pytest.raises(ValueError):
        store.attachment('../../etc/passwd')
    with pytest.raises(KeyError):
        store.attachment('0' * 32)
    store.close()


def test_video_ranges_and_safe_downloads(tmp_path):
    store = ChatStore(tmp_path)
    app = ChatApplication(store, queue.Queue(), Hub())
    attachment = store.upload('clip.mp4', 'video/mp4', base64.b64encode(b'0123456789').decode())
    handler = Handler(attachment['content_url'], headers={'Range': 'bytes=3-6'})
    app.routes(handler)
    assert handler.status == 206
    assert handler.wfile.getvalue() == b'3456'
    assert handler.response_headers['Content-Range'] == 'bytes 3-6/10'
    suffix = Handler(attachment['content_url'], headers={'Range': 'bytes=-2'})
    app.routes(suffix)
    assert suffix.wfile.getvalue() == b'89'
    invalid = Handler(attachment['content_url'], headers={'Range': 'bytes=20-'})
    app.routes(invalid)
    assert invalid.status == 416
    html = store.upload('page.html', 'text/html', base64.b64encode(b'<script/>').decode())
    handler = Handler(html['content_url'])
    app.routes(handler)
    assert handler.response_headers['Content-Type'] == 'application/octet-stream'
    assert handler.response_headers['Content-Disposition'].startswith('attachment;')
    traversal = Handler('/../../etc/passwd')
    assert app.routes(traversal) is True and traversal.status == 404
    store.close()


def test_import_is_idempotent_and_explicit_about_lost_media(tmp_path):
    store = ChatStore(tmp_path)
    with pytest.raises(ValueError):
        store.import_history({'events': ['invalid event']})
    assert store.chats() == []
    chat = store.import_history({'events': [{'type': 'chat', 'from': 'user', 'text': 'old', 'images': 1}]})
    assert store.import_history([])['id'] == chat['id']
    message = store.messages(chat['id'])[0]
    assert message['attachments'] == []
    assert message['metadata']['media_recovered'] is False
    store.close()


def test_bad_selection_does_not_persist_or_enqueue(tmp_path):
    store, inbox = ChatStore(tmp_path), queue.Queue()
    app = ChatApplication(store, inbox, Hub(), [{'id': 'known'}])
    chat = store.create_chat()
    handler = Handler(f"/api/chats/{chat['id']}/messages", 'POST', {'text': 'hello', 'connection_ids': ['unknown']})
    app.routes(handler)
    assert handler.status == 400
    assert store.messages(chat['id']) == [] and inbox.empty()
    store.close()


def test_cross_origin_and_unavailable_connection_rejected(tmp_path):
    store = ChatStore(tmp_path)
    app = ChatApplication(store, queue.Queue(), Hub(), [{'id': 'offline', 'selectable': False}])
    remote = Handler('/say', 'POST', {'text': 'hello'}, headers={'Origin': 'https://example.org', 'Host': '127.0.0.1:8771'})
    app.routes(remote)
    assert remote.status == 403 and store.chats() == []
    chat = store.create_chat()
    with pytest.raises(ValueError):
        app.submit(chat['id'], {'text': 'hello', 'connection_ids': ['offline']})
    store.close()


def test_hub_scopes_live_events_and_frames():
    from collections import deque
    hub = Hub()
    clients = [{'chat_id': ident, 'events': deque(), 'frames': {}} for ident in (None, 'a', 'b')]
    hub.clients.extend(clients)
    hub.publish({'type': 'message', 'chat_id': 'a', 'message': {'id': 'm'}})
    hub.publish({'type': 'frame', 'chat_id': 'b', 'source': 'same', 'data': 'b'})
    hub.publish({'type': 'frame', 'chat_id': 'a', 'source': 'same', 'data': 'a'})
    assert len(clients[0]['events']) == len(clients[1]['events']) == 1
    assert not clients[2]['events']
    assert len(clients[0]['frames']) == 2
    assert next(iter(clients[1]['frames'].values()))['data'] == 'a'


def test_worker_isolates_agents_and_retains_all_media(monkeypatch):
    import threading
    from types import SimpleNamespace
    import tensorcode.agent
    from tensorcode.agent.interpretation import InterpretationWorkspace
    from examples.general_agent.server import worker
    created = []
    class FakeAgent:
        def __init__(self, plugins, reader=None):
            self.grammar = SimpleNamespace()
            self.plugins = plugins
            self.interpretations = InterpretationWorkspace()
            self.received = []
            created.append(self)
        def turn(self, text, images=()):
            self.received.append((text, images))
            return SimpleNamespace(events=[], reply='recorded', seconds=0)
    monkeypatch.setattr(tensorcode.agent, 'Agent', FakeAgent)
    inbox, events = queue.Queue(), queue.Queue()
    attachment = {'id': 'file', 'name': 'notes.pdf', 'media_type': 'application/pdf', 'data': b'pdf', 'size': 3}
    image = {'id': 'image', 'name': 'scene.png', 'media_type': 'image/png', 'data': b'pixels', 'size': 6}
    video = {'id': 'video', 'name': 'clip.mp4', 'media_type': 'video/mp4', 'data': b'video', 'size': 5}
    for chat, attachments in [('a', [attachment, image, video]), ('b', []), ('a', [])]:
        inbox.put({'chat_id': chat, 'message_id': 'message', 'text': chat, 'connection_ids': [], 'attachments': attachments})
    inbox.put(None)
    thread = threading.Thread(target=worker, args=(inbox, events, 10, None, ()))
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(created) == 2
    assert created[0].received == [('a', [b'pixels']), ('a', [])]
    assert created[1].received == [('b', [])]
    sources = created[0].interpretations.sources()
    assert [s.modality for s in sources] == ['file', 'image', 'video']
    assert [s.payload for s in sources] == [b'pdf', b'pixels', b'video']
    assert created[1].interpretations.sources() == ()


def test_long_assistant_reply_preserved_and_later_turns_still_process(tmp_path):
    store, inbox = ChatStore(tmp_path), queue.Queue()
    app = ChatApplication(store, inbox, Hub())
    chat = store.create_chat()
    original = app.submit(chat['id'], {'text': 'x' * 20000})['message']
    reply = 'x' * 20000 + ' — explanation and remaining uncertainty.'
    app.receive({'type': 'assistant_result', 'chat_id': chat['id'], 'message_id': original['id'], 'text': reply})
    app.receive({'type': 'busy', 'chat_id': chat['id'], 'message_id': original['id'], 'busy': False})
    assert store.messages(chat['id'])[1]['text'] == reply
    assert store.chat(chat['id'])['status'] == 'idle'
    following = app.submit(chat['id'], {'text': 'next'})['message']
    app.receive({'type': 'assistant_result', 'chat_id': chat['id'], 'message_id': following['id'], 'text': 'still working'})
    assert store.messages(chat['id'])[-1]['text'] == 'still working'
    with pytest.raises(ValueError, match='20000'):
        app.submit(chat['id'], {'text': 'x' * 20001})
    store.close()


@pytest.mark.parametrize('endpoint', ['api', 'cli'])
def test_omitted_connection_selection_never_mounts_configured_adapters(tmp_path, endpoint):
    store, inbox = ChatStore(tmp_path), queue.Queue()
    app = ChatApplication(store, inbox, Hub(), [{'id': 'browser', 'selectable': True}, {'id': 'gym', 'selectable': True}])
    chat = store.create_chat()
    path = '/say' if endpoint == 'cli' else f"/api/chats/{chat['id']}/messages"
    handler = Handler(path, 'POST', {'text': 'hello', 'chat_id': chat['id']})
    app.routes(handler)
    assert handler.status == 202
    assert handler.json()['message']['connection_ids'] == []
    assert inbox.get_nowait()['connection_ids'] == []
    app.submit(chat['id'], {'text': 'explicit', 'connection_ids': ['gym']})
    assert inbox.get_nowait()['connection_ids'] == ['gym']
    store.close()


def test_same_chat_persistence_and_enqueue_share_order(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    first_waiting, release_first, second_started = threading.Event(), threading.Event(), threading.Event()
    class DelayedQueue(queue.Queue):
        def put(self, item):
            if item['text'] == 'first':
                first_waiting.set()
                assert release_first.wait(3)
            super().put(item)
    store, inbox = ChatStore(tmp_path), DelayedQueue()
    app = ChatApplication(store, inbox, Hub())
    chat = store.create_chat()
    def second():
        second_started.set()
        return app.submit(chat['id'], {'text': 'second'})
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(app.submit, chat['id'], {'text': 'first'})
        try:
            assert first_waiting.wait(3)
            second_future = executor.submit(second)
            assert second_started.wait(3)
            # While the first request is persisted but not yet queued, another
            # same-chat request must not slip through and reach the worker first.
            with pytest.raises(queue.Empty):
                inbox.get(timeout=0.1)
            assert [m['text'] for m in store.messages(chat['id'])] == ['first']
        finally:
            release_first.set()
        first_future.result(timeout=3)
        second_future.result(timeout=3)
    persisted = [m['id'] for m in store.messages(chat['id'])]
    enqueued = [inbox.get_nowait()['message_id'], inbox.get_nowait()['message_id']]
    assert enqueued == persisted
    store.close()
