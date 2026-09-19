"""Bounded HTTP transport for persistent chat; no static filesystem fallback."""
from __future__ import annotations
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, quote
from examples.general_agent.chat_store import MAX_REQUEST_BYTES

PAGE = Path(__file__).with_name('chat.html')


class ChatApplication:
    def __init__(self, store, inbox, hub, connections=()):
        self.store, self.inbox, self.hub = store, inbox, hub
        self.connections = list(connections)

    def submit(self, chat_id, body, origin='ui'):
        text, attachments = body.get('text', ''), body.get('attachment_ids', [])
        ids = body.get('connection_ids', [c['id'] for c in self.connections if c.get('selectable', True)])
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)) or any(i not in {c['id'] for c in self.connections if c.get('selectable', True)} for i in ids):
            raise ValueError('unknown or duplicate connection id')
        if not text and not attachments:
            raise ValueError('a message requires text or attachments')
        message = self.store.add_message(chat_id, 'user', text, attachment_ids=attachments, connection_ids=ids, origin=origin, status='queued')
        self.hub.publish({'type': 'message', 'chat_id': chat_id, 'message': message})
        self.inbox.put({'chat_id': chat_id, 'message_id': message['id'], 'text': text,
                        'connection_ids': ids, 'attachments': [self.store.attachment(a['id'], content=True) for a in message['attachments']]})
        return {'message': message, 'queued': True, 'chat_id': chat_id}

    def receive(self, event):
        chat_id, message_id = event.get('chat_id'), event.get('message_id')
        if event['type'] == 'assistant_result':
            message = self.store.add_message(chat_id, 'assistant', event['text'], origin='agent', status='error' if event.get('error') else 'completed', metadata={'reply_to': message_id, 'seconds': event.get('seconds')})
            self.hub.publish({'type': 'message', 'chat_id': chat_id, 'message': message})
            return
        if event['type'] == 'busy':
            status = 'running' if event['busy'] else 'error' if event.get('error') else 'completed'
            self.store.status(chat_id, message_id, status)
            self.hub.publish({'type': 'message', 'chat_id': chat_id, 'message': self.store.message(message_id)})
        self.hub.publish(event)

    @staticmethod
    def response(handler, value, status=200):
        data = json.dumps(value).encode()
        handler.send_response(status)
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @staticmethod
    def read_body(handler):
        size = int(handler.headers.get('Content-Length') or 0)
        if size < 0 or size > MAX_REQUEST_BYTES:
            raise ValueError('request exceeds upload limit')
        body = json.loads(handler.rfile.read(size) or b'{}')
        if not isinstance(body, dict):
            raise ValueError('JSON object required')
        return body

    def routes(self, handler):
        url = urlsplit(handler.path)
        parts = url.path.strip('/').split('/')
        try:
            if handler.command == 'POST':
                origin = handler.headers.get('Origin')
                if origin and origin != 'http://' + handler.headers.get('Host', ''):
                    self.response(handler, {'error': 'cross-origin writes are not allowed'}, 403)
                    return True
                if handler.headers.get('Sec-Fetch-Site') == 'cross-site':
                    self.response(handler, {'error': 'cross-site writes are not allowed'}, 403)
                    return True
            if url.path == '/' and handler.command == 'GET':
                data = PAGE.read_bytes()
                handler.send_response(200)
                handler.send_header('Content-Type', 'text/html; charset=utf-8')
                handler.send_header('Content-Length', str(len(data)))
                handler.end_headers()
                handler.wfile.write(data)
            elif url.path == '/events' and handler.command == 'GET':
                chat_id = parse_qs(url.query).get('chat_id', [None])[0]
                if chat_id:
                    self.store.chat(chat_id)
                self.hub.serve(handler, chat_id=chat_id)
            elif url.path == '/api/chats':
                if handler.command == 'GET':
                    self.response(handler, {'chats': self.store.chats()})
                elif handler.command == 'POST':
                    self.response(handler, {'chat': self.store.create_chat(self.read_body(handler).get('title', 'New chat'))}, 201)
                else:
                    self.response(handler, {'error': 'method not allowed'}, 405)
            elif len(parts) == 3 and parts[:2] == ['api', 'chats'] and handler.command == 'GET':
                self.response(handler, {'chat': self.store.chat(parts[2]), 'messages': self.store.messages(parts[2])})
            elif len(parts) == 4 and parts[:2] == ['api', 'chats'] and parts[3] == 'messages' and handler.command == 'POST':
                self.response(handler, self.submit(parts[2], self.read_body(handler)), 202)
            elif url.path == '/say' and handler.command == 'POST':
                body = self.read_body(handler)
                images = body.pop('images', [])
                if not isinstance(images, list) or len(images) > 16:
                    raise ValueError('images must contain at most 16 base64 images')
                attachment_ids = body.get('attachment_ids', [])
                if not isinstance(attachment_ids, list):
                    raise ValueError('attachment_ids must be a list')
                for index, image in enumerate(images):
                    attachment_ids.append(self.store.upload(f'cli-image-{index + 1}.png', 'image/png', image)['id'])
                body['attachment_ids'] = attachment_ids
                chat_id = body.get('chat_id') or self.store.cli_chat()['id']
                self.response(handler, {'ok': True, **self.submit(chat_id, body, 'cli')}, 202)
            elif url.path == '/api/attachments' and handler.command == 'POST':
                body = self.read_body(handler)
                attachment = self.store.upload(body.get('name'), body.get('media_type', 'application/octet-stream'), body.get('data'))
                self.response(handler, {'attachment': attachment}, 201)
            elif len(parts) == 4 and parts[:2] == ['api', 'attachments'] and parts[3] == 'content' and handler.command == 'GET':
                self.serve_attachment(handler, self.store.attachment(parts[2], content=True))
            elif url.path == '/api/connections' and handler.command == 'GET':
                self.response(handler, {'connections': self.connections})
            else:
                self.response(handler, {'error': 'not found'}, 404)
        except KeyError as exc:
            self.response(handler, {'error': str(exc)}, 404)
        except (ValueError, TypeError) as exc:
            self.response(handler, {'error': str(exc)}, 400)
        return True

    def serve_attachment(self, handler, attachment):
        data, total = attachment['data'], attachment['size']
        start, end, status = 0, total - 1, 200
        requested = handler.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()) or total == 0:
                self.response(handler, {'error': 'unsatisfiable byte range'}, 416)
                return
            first, last = match.groups()
            if first:
                start, end = int(first), min(int(last) if last else total - 1, total - 1)
            else:
                start, end = max(0, total - int(last)), total - 1
            if start > end or start >= total:
                self.response(handler, {'error': 'unsatisfiable byte range'}, 416)
                return
            status = 206
        handler.send_response(status)
        media = attachment['media_type']
        inline = media.startswith(('image/', 'video/', 'audio/')) and media != 'image/svg+xml'
        handler.send_header('Content-Type', media if inline else 'application/octet-stream')
        handler.send_header('X-Content-Type-Options', 'nosniff')
        handler.send_header('Content-Security-Policy', 'sandbox')
        handler.send_header('Content-Disposition', ('inline' if inline else 'attachment') + "; filename*=UTF-8''" + quote(attachment['name'], safe=''))
        handler.send_header('Accept-Ranges', 'bytes')
        if status == 206:
            handler.send_header('Content-Range', f'bytes {start}-{end}/{total}')
        result = data[start:end + 1]
        handler.send_header('Content-Length', str(len(result)))
        handler.end_headers()
        handler.wfile.write(result)
