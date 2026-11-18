// Exhaustive check of fxp_add and fxp_mul: all 65,536 int8 input pairs.
`include "config.svh"
`timescale 1ns/1ps
module tb_fxp_ops;
    logic signed [7:0] a, b, y_add, y_mul;
    logic [7:0] add_exp[65536];
    logic [7:0] mul_exp[65536];
    int errors = 0;

    fxp_add #(.N(8)) u_add (.a(a), .b(b), .y(y_add));
    fxp_mul #(.N(8), .F(`FXP_FRAC)) u_mul (.a(a), .b(b), .y(y_mul));

    initial begin
        $readmemh("vectors/fxp_add_y.hex", add_exp);
        $readmemh("vectors/fxp_mul_y.hex", mul_exp);
        for (int ia = 0; ia < 256; ia++)
            for (int ib = 0; ib < 256; ib++) begin
                a = ia - 128;
                b = ib - 128;
                #1;
                if (y_add !== add_exp[ia*256 + ib] || y_mul !== mul_exp[ia*256 + ib]) begin
                    if (errors < 10)
                        $display("MISMATCH a=%0d b=%0d: add %0d/%0d mul %0d/%0d", a, b,
                                 y_add, $signed(add_exp[ia*256 + ib]),
                                 y_mul, $signed(mul_exp[ia*256 + ib]));
                    errors++;
                end
            end
        if (errors == 0) $display("PASS tb_fxp_ops: 65536 input pairs for add and mul");
        else $display("FAIL tb_fxp_ops: %0d mismatches", errors);
        $finish;
    end
endmodule
