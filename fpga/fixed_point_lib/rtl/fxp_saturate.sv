`include "fxp_pkg.sv"

module fxp_saturate #(
    parameter int N = 8
)(
    input  logic signed [31:0] in,
    output logic signed [N-1:0] out
);
    import fxp_pkg::*;
    always_comb out = sat_signed#(N)(in);
endmodule
