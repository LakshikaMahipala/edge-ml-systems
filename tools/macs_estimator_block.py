import argparse

def dwconv1d_macs(C: int, L: int, K: int) -> int:
    Lout = L - K + 1
    return C * Lout * K

def pointwise_macs(Cin: int, Cout: int, Lout: int) -> int:
    # 1x1 conv per time step
    return Lout * Cin * Cout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--Cout", type=int, default=4)
    args = ap.parse_args()

    Lout = args.L - args.K + 1
    mac_dw = dwconv1d_macs(args.C, args.L, args.K)
    mac_pw = pointwise_macs(args.C, args.Cout, Lout)

    print("MobileNet-like block MAC estimator")
    print(f"C={args.C} L={args.L} K={args.K} Lout={Lout} Cout={args.Cout}")
    print(f"DWConv1D MACs: {mac_dw}")
    print(f"Pointwise (1x1) MACs: {mac_pw} (future)")
    print(f"Total (DW + PW): {mac_dw + mac_pw}")

if __name__ == "__main__":
    main()
