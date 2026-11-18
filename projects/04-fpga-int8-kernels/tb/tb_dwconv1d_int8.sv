// Checks dwconv1d_int8 against golden vectors, and its cycle count.
`include "config.svh"
`timescale 1ns/1ps
module tb_dwconv1d_int8;
    localparam int C = `DW_C, L = `DW_L, K = `DW_K, N = `N_TESTS;
    localparam int LOUT = L - K + 1;
    localparam int EXPECTED_CYCLES = C * LOUT * (K + 1);

    logic clk = 0, rst = 1, start = 0, busy, done;
    logic [C*L*8-1:0]    x_flat;
    logic [C*LOUT*8-1:0] y_flat;
    logic [7:0] x_mem[N*C*L];
    logic [7:0] y_exp[N*C*LOUT];
    int errors = 0, cycles;
    time t_start;

    always #5 clk = ~clk;

    dwconv1d_int8 #(.C(C), .L(L), .K(K), .SHIFT(`DW_SHIFT)) dut (
        .clk(clk), .rst(rst), .start(start), .x_flat(x_flat),
        .busy(busy), .done(done), .y_flat(y_flat));

    initial begin
        $readmemh("vectors/dw_x.hex", x_mem);
        $readmemh("vectors/dw_y.hex", y_exp);
        repeat (2) @(posedge clk);
        rst <= 0;
        for (int t = 0; t < N; t++) begin
            for (int e = 0; e < C*L; e++) x_flat[8*e +: 8] = x_mem[t*C*L + e];
            @(posedge clk) start <= 1;
            @(posedge clk) start <= 0;  // the DUT samples start on this edge
            t_start = $time;
            @(posedge done);
            cycles = ($time - t_start) / 10;  // clock period = 10 ns
            #1;                                     // let all outputs settle
            if (cycles != EXPECTED_CYCLES) begin
                $display("LATENCY test %0d: %0d cycles, expected %0d", t, cycles, EXPECTED_CYCLES);
                errors++;
            end
            for (int e = 0; e < C*LOUT; e++)
                if (y_flat[8*e +: 8] !== y_exp[t*C*LOUT + e]) begin
                    if (errors < 10)
                        $display("MISMATCH test %0d elem %0d: got %0d expected %0d", t, e,
                                 $signed(y_flat[8*e +: 8]), $signed(y_exp[t*C*LOUT + e]));
                    errors++;
                end
        end
        if (errors == 0)
            $display("PASS tb_dwconv1d_int8: %0d vectors, %0d cycles each", N, EXPECTED_CYCLES);
        else $display("FAIL tb_dwconv1d_int8: %0d errors", errors);
        $finish;
    end
endmodule
