import argparse
import time
import serial  # pyserial required later

from protocol import Frame, TYPE_INPUT, TYPE_PING, decode_stream, TYPE_OUTPUT, TYPE_PONG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--mode", type=str, default="ping", choices=["ping", "send_vector"])
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.1)

    if args.mode == "ping":
        f = Frame(typ=TYPE_PING, payload=b"").encode()
        ser.write(f)
        time.sleep(0.1)
        buf = ser.read(4096)
        frames, _ = decode_stream(buf)
        ok = any(fr.typ == TYPE_PONG for fr in frames)
        print("PONG" if ok else "NO PONG")

    if args.mode == "send_vector":
        # v0: send 8 int8 values
        vec = bytes([10 & 0xFF, 0xFD, 7, 2, 0xF8, 1, 4, 0xFE])  # example signed bytes
        f = Frame(typ=TYPE_INPUT, payload=vec).encode()
        ser.write(f)
        time.sleep(0.1)
        buf = ser.read(4096)
        frames, _ = decode_stream(buf)
        outs = [fr for fr in frames if fr.typ == TYPE_OUTPUT]
        if outs:
            print("OUTPUT:", list(outs[0].payload))
        else:
            print("NO OUTPUT")

if __name__ == "__main__":
    main()
