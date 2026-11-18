// Checks int8_fc against golden vectors from golden/gen_vectors.py.
`include "config.svh"
`timescale 1ns/1ps
module tb_int8_fc;
    localparam int IN = `FC_IN, OUT = `FC_OUT, N = `N_TESTS;

    logic [IN*8-1:0]  x_flat;
    logic [OUT*8-1:0] y_flat;
    logic [7:0] x_mem[N*IN];
    logic [7:0] y_exp[N*OUT];
    int errors = 0;

    int8_fc #(.IN(IN), .OUT(OUT), .SHIFT(`FC_SHIFT)) dut (.x_flat(x_flat), .y_flat(y_flat));

    initial begin
        $readmemh("vectors/fc_x.hex", x_mem);
        $readmemh("vectors/fc_y.hex", y_exp);
        for (int t = 0; t < N; t++) begin
            for (int i = 0; i < IN; i++) x_flat[8*i +: 8] = x_mem[t*IN + i];
            #1;
            for (int j = 0; j < OUT; j++)
                if (y_flat[8*j +: 8] !== y_exp[t*OUT + j]) begin
                    if (errors < 10)
                        $display("MISMATCH test %0d out %0d: got %0d expected %0d", t, j,
                                  $signed(y_flat[8*j +: 8]), $signed(y_exp[t*OUT + j]));
                    errors++;
                end
        end
        if (errors == 0) $display("PASS tb_int8_fc: %0d vectors x %0d outputs", N, OUT);
        else $display("FAIL tb_int8_fc: %0d mismatches", errors);
        $finish;
    end
endmodule
