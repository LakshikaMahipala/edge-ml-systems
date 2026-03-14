Mini-project 4: Real-time Video Classifier 

Goal
- Run a live webcam/video classifier and overlay latency + FPS.

Run (later)
pip install -r requirements.txt
python -m src.app --source 0 --model resnet18 --device cpu

Keys
- Press 'q' to quit

Notes
- This is a host-side pipeline.
- FPGA offload point is documented in docs/pipeline_diagram.md
