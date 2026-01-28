`include "fxp_pkg.sv"

module fxp_mul #(
    parameter int N = 8,
    parameter int F = 7
)(
    input  logic signed [N-1:0] a,
    input  logic signed [N-1:0] b,
    output logic signed [N-1:0] y
);
    import fxp_pkg::*;
    logic signed [31:0] prod_full;
    logic signed [31:0] prod_q;

    always_comb begin
        prod_full = $signed(a) * $signed(b);          // has 2F fractional bits
        prod_q    = rshift_round(prod_full, F);       // back to F fractional bits
        y         = sat_signed#(N)(prod_q);
    end
endmodule
