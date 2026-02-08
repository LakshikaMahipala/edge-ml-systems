import argparse

def conv_macs(h_out, w_out, c_out, k, c_in):
    return h_out * w_out * c_out * (k * k * c_in)

def fc_macs(c_in, c_out):
    return c_in * c_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=3.0, help="cycles per MAC proxy factor")
    args = ap.parse_args()

    # Example for 28x28 input:
    # Conv1: input 28x28x1 -> 28x28x8, k=3
    mac1 = conv_macs(28, 28, 8, 3, 1)
    # Pool -> 14x14x8
    # Conv2: 14x14x8 -> 14x14x16, k=3
    mac2 = conv_macs(14, 14, 16, 3, 8)
    # Pool -> 7x7x16 => flatten 784
    # FC1: 784 -> 32
    mac3 = fc_macs(7*7*16, 32)
    # FC2: 32 -> 10 (example)
    mac4 = fc_macs(32, 10)

    total = mac1 + mac2 + mac3 + mac4
    cycles = total * args.alpha

    print("Tiny CNN MAC/cycle proxy report")
    print(f"conv1_macs={mac1}")
    print(f"conv2_macs={mac2}")
    print(f"fc1_macs={mac3}")
    print(f"fc2_macs={mac4}")
    print(f"total_macs={total}")
    print(f"alpha={args.alpha} cycles_per_mac_proxy")
    print(f"estimated_cycles={int(cycles)}")

if __name__ == "__main__":
    main()
