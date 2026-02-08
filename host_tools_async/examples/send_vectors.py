import argparse
import os
import random
from host_tools_async.uart_async_client import UARTAsyncClient
from host_tools_async.metrics import LatencyStats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--in", dest="in_dim", type=int, default=8)
    ap.add_argument("--timeout_s", type=float, default=0.5)
    args = ap.parse_args()

    cli = UARTAsyncClient(args.port, baud=args.baud)
    stats = LatencyStats(max_hist=2000)

    try:
        for k in range(args.n):
            req_id = k % 256
            # random int8 vector
            xs = bytes([(random.randint(-128, 127) & 0xFF) for _ in range(args.in_dim)])
            out = cli.request_vector(req_id=req_id, x_int8=xs, timeout_s=args.timeout_s)
            if out is None:
                print("timeout", k)
                continue
            _, y, lat_ms = out
            stats.add(lat_ms)
            if (k+1) % 10 == 0:
                print("k", k+1, "lat", stats.summary(), "cli", cli.stats)
    finally:
        cli.close()

if __name__ == "__main__":
    main()
