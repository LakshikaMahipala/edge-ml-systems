FPGA Cost Model v1 

What this adds over v0
- explicit variant types (BASE vs LOWRANK)
- transform overhead term (placeholder for Winograd/FFT/SVD intermediates)
- BRAM proxy

Primary use
- ranking design points
- end-to-end budgeting with IO included (UART today)
- later calibration with measured FPGA data

Run later
python src/compare_variants.py --IN 1024 --OUT 1024 --ranks 64,128,256,512
