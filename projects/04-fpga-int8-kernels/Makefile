# INT8 FPGA kernels: golden model, test vectors and RTL simulation (Icarus Verilog).
#   make test     -> Python golden-model unit tests
#   make vectors  -> generate test vectors + expected outputs
#   make sim      -> simulate all testbenches (needs iverilog >= 11)
#   make all      -> everything

PYTHON  ?= python3
IVERILOG = iverilog -g2012 -Wall -I vectors
BUILD    = build
PKG      = rtl/fxp_pkg.sv
TBS      = tb_fxp_ops tb_int8_fc tb_int8_fc_seq tb_dwconv1d_int8

tb_fxp_ops_SRC       = rtl/fxp_ops.sv
tb_int8_fc_SRC       = rtl/int8_fc.sv
tb_int8_fc_seq_SRC   = rtl/int8_fc_seq.sv
tb_dwconv1d_int8_SRC = rtl/dwconv1d_int8.sv

.PHONY: all test vectors sim clean
all: test sim

test:
	cd golden && $(PYTHON) -m pytest -q

vectors/config.svh: golden/gen_vectors.py golden/int8_ref.py
	cd golden && $(PYTHON) gen_vectors.py --out ../vectors

vectors: vectors/config.svh

sim: vectors
	@mkdir -p $(BUILD)
	@set -e; for tb in $(TBS); do \
	  case $$tb in \
	    tb_fxp_ops) src="$(tb_fxp_ops_SRC)";; \
	    tb_int8_fc) src="$(tb_int8_fc_SRC)";; \
	    tb_int8_fc_seq) src="$(tb_int8_fc_seq_SRC)";; \
	    tb_dwconv1d_int8) src="$(tb_dwconv1d_int8_SRC)";; \
	  esac; \
	  $(IVERILOG) -o $(BUILD)/$$tb.vvp $(PKG) $$src tb/$$tb.sv; \
	  vvp -n $(BUILD)/$$tb.vvp | tee $(BUILD)/$$tb.log; \
	  grep -q "^PASS" $(BUILD)/$$tb.log; \
	done
	@echo "All RTL testbenches passed."

clean:
	rm -rf $(BUILD) vectors
