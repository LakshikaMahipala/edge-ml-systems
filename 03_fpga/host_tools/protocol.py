from __future__ import annotations
from dataclasses import dataclass
from typing import List

SOF = 0xA5

TYPE_INPUT = 0x01
TYPE_OUTPUT = 0x02
TYPE_PING = 0x7F
TYPE_PONG = 0x80

def crc8(data: bytes) -> int:
    c = 0x00
    for b in data:
        for i in range(8):
            fb = ((c >> 7) & 1) ^ ((b >> (7 - i)) & 1)
            c = ((c << 1) & 0xFF)
            if fb:
                c ^= 0x07
    return c

@dataclass
class Frame:
    typ: int
    payload: bytes

    def encode(self) -> bytes:
        ln = len(self.payload)
        header = bytes([ln & 0xFF, self.typ & 0xFF])
        c = crc8(header + self.payload)
        return bytes([SOF]) + header + self.payload + bytes([c])

def decode_stream(buf: bytes) -> tuple[List[Frame], bytes]:
    """
    Decode as many frames as possible from buf.
    Returns (frames, remainder).
    """
    frames: List[Frame] = []
    i = 0
    while True:
        # find SOF
        j = buf.find(bytes([SOF]), i)
        if j < 0:
            return frames, b""
        if j + 3 > len(buf):
            return frames, buf[j:]  # partial header
        ln = buf[j + 1]
        typ = buf[j + 2]
        total = 1 + 1 + 1 + ln + 1
        if j + total > len(buf):
            return frames, buf[j:]  # partial frame
        payload = buf[j + 3 : j + 3 + ln]
        crc_rx = buf[j + 3 + ln]
        crc_calc = crc8(bytes([ln, typ]) + payload)
        if crc_rx == crc_calc:
            frames.append(Frame(typ=typ, payload=payload))
            i = j + total
        else:
            # bad frame; resync by searching next byte
            i = j + 1
