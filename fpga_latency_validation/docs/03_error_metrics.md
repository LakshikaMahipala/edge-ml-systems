# Error metrics (secondary)

Besides ranking, we track error:
- Absolute error: |pred - meas|
- Relative error: |pred - meas| / meas
- MAPE: mean absolute percentage error

For FPGA early-stage research:
- ranking matters most for NAS
- absolute error matters later for deployment budgeting
