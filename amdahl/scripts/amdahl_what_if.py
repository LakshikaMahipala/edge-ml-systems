import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_pre_ms", type=float, required=True)
    ap.add_argument("--t_inf_ms", type=float, required=True)
    ap.add_argument("--t_post_ms", type=float, required=True)
    ap.add_argument("--t_copy_ms", type=float, default=0.0)

    ap.add_argument("--target", type=str, required=True, choices=["pre", "inf", "post", "copy"])
    ap.add_argument("--S", type=float, required=True, help="speedup factor applied to target stage")
    args = ap.parse_args()

    t_pre = args.t_pre_ms
    t_inf = args.t_inf_ms
    t_post = args.t_post_ms
    t_copy = args.t_copy_ms

    t_total = t_pre + t_inf + t_post + t_copy

    if args.target == "pre":
        t_pre /= args.S
    elif args.target == "inf":
        t_inf /= args.S
    elif args.target == "post":
        t_post /= args.S
    else:
        t_copy /= args.S

    t_new = t_pre + t_inf + t_post + t_copy
    sp = t_total / t_new

    print("What-if speedup")
    print(f"Original total (ms): {t_total:.3f}")
    print(f"New total (ms):      {t_new:.3f}")
    print(f"Speedup:             {sp:.3f}x")
    print(f"New stage times (ms): pre={t_pre:.3f} inf={t_inf:.3f} post={t_post:.3f} copy={t_copy:.3f}")

if __name__ == "__main__":
    main()
