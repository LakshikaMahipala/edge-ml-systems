`include "fxp_pkg.sv"

module fxp_add #(
    parameter int N = 8
)(
    input  logic signed [N-1:0] a,
    input  logic signed [N-1:0] b,
    output logic signed [N-1:0] y
);
    import fxp_pkg::*;
    logic signed [31:0] sum;
    always_comb begin
        sum = $signed(a) + $signed(b);
        y   = sat_signed#(N)(sum);
    end
endmodule
