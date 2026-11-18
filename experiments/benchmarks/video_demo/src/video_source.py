import cv2

def open_source(source: str):
    """
    source:
      - "0" for webcam index 0
      - path to a video file
    """
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    return cap
