import argparse
import cv2

from video_source import open_source
from preprocess import preprocess_bgr_to_nchw_float
from infer_pytorch import PyTorchVideoInfer
from overlay import PerfOverlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0", help='webcam index like "0" or video path')
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18","mobilenet_v3_small","efficientnet_b0"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--out", type=str, default="", help="optional output video path")
    args = ap.parse_args()

    cap = open_source(args.source)
    infer = PyTorchVideoInfer(model_name=args.model, device=args.device)
    overlay = PerfOverlay()

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        x = preprocess_bgr_to_nchw_float(frame, args.input_size)
        top1, conf, lat_ms = infer.infer(x)
        frame = overlay.draw(frame, top1, conf, lat_ms)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("video_demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
