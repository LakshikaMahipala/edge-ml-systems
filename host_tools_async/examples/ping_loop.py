import argparse
import time
from host_tools_async.uart_async_client import UARTAsyncClient

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=str, required=True)
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    cli = UARTAsyncClient(args.port, baud=args.baud)
    try:
        for i in range(20):
            ok = cli.ping(timeout_s=0.5)
            print(i, "PONG" if ok else "NO PONG", cli.stats)
            time.sleep(0.2)
    finally:
        cli.close()

if __name__ == "__main__":
    main()
