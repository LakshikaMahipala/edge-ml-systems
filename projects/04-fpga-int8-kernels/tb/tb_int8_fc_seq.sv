// Checks int8_fc_seq against golden vectors, and that latency is IN + 1 cycles.
`include "config.svh"
`timescale 1ns/1ps
module tb_int8_fc_seq;
    localparam int IN = `FC_IN, OUT = `FC_OUT, N = `N_TESTS;

    logic clk = 0, rst = 1, start = 0, busy, done;
    logic [IN*8-1:0]  x_flat;
    logic [OUT*8-1:0] y_flat;
    logic [7:0] x_mem[N*IN];
    logic [7:0] y_exp[N*OUT];
    int errors = 0, cycles;
    time t_start;

    always #5 clk = ~clk;

    int8_fc_seq #(.IN(IN), .OUT(OUT), .SHIFT(`FC_SHIFT)) dut (
        .clk(clk), .rst(rst), .start(start), .x_flat(x_flat),
        .busy(busy), .done(done), .y_flat(y_flat));

    initial begin
        $readmemh("vectors/fc_x.hex", x_mem);
        $readmemh("vectors/fc_y.hex", y_exp);
        repeat (2) @(posedge clk);
        rst <= 0;
        for (int t = 0; t < N; t++) begin
            for (int i = 0; i < IN; i++) x_flat[8*i +: 8] = x_mem[t*IN + i];
            @(posedge clk) start <= 1;
            @(posedge clk) start <= 0;  // the DUT samples start on this edge
            t_start = $time;
            @(posedge done);
            cycles = ($time - t_start) / 10;  // clock period = 10 ns
            #1;                                     // let all outputs settle
            if (cycles != IN + 1) begin
                $display("LATENCY test %0d: %0d cycles, expected %0d", t, cycles, IN + 1);
                errors++;
            end
            for (int j = 0; j < OUT; j++)
                if (y_flat[8*j +: 8] !== y_exp[t*OUT + j]) begin
                    if (errors < 10)
                        $display("MISMATCH test %0d out %0d: got %0d expected %0d", t, j,
                                 $signed(y_flat[8*j +: 8]), $signed(y_exp[t*OUT + j]));
                    errors++;
                end
        end
        if (errors == 0) $display("PASS tb_int8_fc_seq: %0d vectors, %0d cycles each", N, IN + 1);
        else $display("FAIL tb_int8_fc_seq: %0d errors", errors);
        $finish;
    end
endmodule
