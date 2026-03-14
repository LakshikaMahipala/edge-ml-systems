`include "../../fixed_point_lib/rtl/fxp_pkg.sv"

module int8_requant #(
    parameter int SHIFT = 7
)(
    input  logic signed [31:0] acc,
    output logic signed [7:0]  y
);
    import fxp_pkg::*;
    logic signed [31:0] q;

    always_comb begin
        q = rshift_round(acc, SHIFT);
        y = sat_signed#(8)(q);
    end
endmodule
