UART Frame Protocol 

Frame
SOF  LEN  TYPE  PAYLOAD...  CRC8
A5   1B   1B    LEN bytes   1B

CRC8 covers: LEN, TYPE, PAYLOAD (NOT SOF)

Types
0x01 INPUT_VECTOR (host->fpga): payload = IN int8 bytes
0x02 OUTPUT_VECTOR (fpga->host): payload = OUT int8 bytes
0x7F PING (host->fpga): payload empty
0x80 PONG (fpga->host): payload empty

Resync
Receiver scans for SOF (0xA5). If CRC fails, drop frame and keep scanning.

Why this protocol
- Minimal (easy to debug in hex)
- Robust (length + crc)
- Extensible (TYPE)
