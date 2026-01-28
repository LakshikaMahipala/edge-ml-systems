`include "fxp_pkg.sv"

module fxp_scale_shift #(
    parameter int NOUT = 8,
    parameter int SHIFT = 7
)(
    input  logic signed [31:0] in,
    output logic signed [NOUT-1:0] out
);
    import fxp_pkg::*;
    logic signed [31:0] q;

    always_comb begin
        q   = rshift_round(in, SHIFT);
        out = sat_signed#(NOUT)(q);
    end
endmodule
