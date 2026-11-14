"""Bounded JSON frames over the authenticated SSH standard streams."""
import base64
import json
import struct

LIMIT = 128 * 1024 * 1024


class Wire:
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer
        self.sent = self.received = 0

    def _read(self, size):
        pieces = []
        while size:
            p = self.reader.read(size)
            if not p:
                raise EOFError('SSH peer closed the protocol stream')
            pieces.append(p)
            size -= len(p)
        return b''.join(pieces)

    def send(self, message):
        data = json.dumps(message, separators=(',', ':'), allow_nan=False).encode()
        if len(data) > LIMIT:
            raise ValueError('Frame too large')
        self.writer.write(struct.pack('!I', len(data)) + data)
        self.writer.flush()
        self.sent += len(data) + 4

    def recv(self):
        n, = struct.unpack('!I', self._read(4))
        if n > LIMIT:
            raise ValueError('Frame too large')
        data = json.loads(self._read(n))
        self.received += n + 4
        return data


def pack(data):
    return base64.b64encode(data).decode('ascii')


def unpack(data):
    return base64.b64decode(data, validate=True)

