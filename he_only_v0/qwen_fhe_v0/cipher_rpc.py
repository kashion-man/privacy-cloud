"""Bounded local framing for public metadata and serialized ciphertexts only."""
import json
import struct

MAX_FRAME = 64 * 1024 * 1024


def receive_exact(connection, count):
    result = bytearray()
    while len(result) < count:
        part = connection.recv(count-len(result))
        if not part:
            raise EOFError('Ciphertext service disconnected')
        result.extend(part)
    return bytes(result)


def send_frame(connection, frame):
    if len(frame) > MAX_FRAME:
        raise ValueError('Ciphertext frame exceeds public size limit')
    connection.sendall(struct.pack('!I', len(frame))+frame)


def receive_frame(connection):
    count = struct.unpack('!I', receive_exact(connection, 4))[0]
    if count > MAX_FRAME:
        raise ValueError('Ciphertext frame exceeds public size limit')
    return receive_exact(connection, count)


def send_metadata(connection, value):
    frame = json.dumps(value).encode('ascii')
    if len(frame) > 4096:
        raise ValueError('Public metadata exceeds size limit')
    send_frame(connection, frame)


def receive_metadata(connection):
    frame = receive_frame(connection)
    if len(frame) > 4096:
        raise ValueError('Public metadata exceeds size limit')
    return json.loads(frame)
