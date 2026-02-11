# FFT convolution notes (fill after runs)

Machine:
Python/NumPy FFT backend:

Observed crossover:
- For each N, smallest K where FFT beats naive:

Interpretation:
- padding to pow2 helped/hurt?
- when K small, transform overhead dominates
- when K large, FFT wins

FPGA angle:
- large 1D signals are the right target class
- identify buffering + scaling needs for a streaming FFT pipeline
