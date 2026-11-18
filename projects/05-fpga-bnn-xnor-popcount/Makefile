# make test -> Python reference tests | make sim -> RTL simulation (iverilog >= 11)
PYTHON ?= python3
.PHONY: all test sim clean
all: test sim
test:
	cd golden && $(PYTHON) -m pytest -q
vectors/config.svh: golden/gen_vectors.py golden/bnn_ref.py
	cd golden && $(PYTHON) gen_vectors.py --out ../vectors
sim: vectors/config.svh
	@mkdir -p build
	iverilog -g2012 -I vectors -o build/tb_bnn_dot.vvp rtl/popcount32.v rtl/xnor_popcount_dot.v rtl/bnn_dot_top.v tb/tb_bnn_dot.sv
	vvp -n build/tb_bnn_dot.vvp | tee build/tb_bnn_dot.log
	@grep -q "^PASS" build/tb_bnn_dot.log
clean:
	rm -rf build vectors
