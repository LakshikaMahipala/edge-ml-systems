Host async UART client 

Goal
- Non-blocking UART reads + request/response pairing + back-pressure-safe queues.

Key idea
- Reader thread continuously drains serial port to avoid buffer overflow.
- Frames are decoded and routed to per-request queues via req_id.

Protocol note
- INPUT payload:  [req_id][IN bytes]
- OUTPUT payload: [req_id][OUT bytes]

Run later
pip install pyserial
python examples/ping_loop.py --port /dev/ttyUSB0
python examples/send_vectors.py --port /dev/ttyUSB0
