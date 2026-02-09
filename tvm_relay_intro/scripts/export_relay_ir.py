from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    import torch
    import torchvision.models as models

    import tvm
    from tvm import relay

    # 1) Load model
    if args.model == "resnet18":
        net = models.resnet18(weights=None)
    else:
        net = models.mobilenet_v3_small(weights=None)

    net.eval()

    # 2) Trace
    x = torch.randn(1, 3, args.input_size, args.input_size)
    traced = torch.jit.trace(net, x).eval()

    # 3) Convert to Relay
    input_name = "input0"
    shape_list = [(input_name, x.shape)]
    mod, params = relay.frontend.from_pytorch(traced, shape_list)

    # 4) Print + save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    relay_txt = mod.astext(show_meta_data=False)
    out_path = out_dir / f"relay_{args.model}_{args.input_size}.txt"
    out_path.write_text(relay_txt, encoding="utf-8")

    print("=== Relay IR ===")
    print(relay_txt[:4000])  # print first part to keep terminal readable
    print("\nSaved:", out_path)

    # save params count (quick sanity)
    pcount = sum(int(v.size) for v in params.values())
    (out_dir / f"params_count_{args.model}_{args.input_size}.txt").write_text(
        f"param_tensors={len(params)}\nparam_total_elems={pcount}\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
