// Streams random packed vectors into the XNOR-popcount core and checks every
// result against golden/gen_vectors.py. Inputs arrive with random one-cycle
// gaps (word_valid low) and `start` coincides with the first word, to test
// both stalls and back-to-back operation.
`include "config.svh"
`timescale 1ns/1ps
module tb_bnn_dot;
    localparam int N_BITS = `N_BITS, WORD_W = `WORD_W, N_WORDS = N_BITS / WORD_W;
    localparam int N = `N_TESTS;

    logic clk = 0, rst_n = 0;
    logic start = 0, word_valid = 0, last_word = 0;
    logic [WORD_W-1:0] a_word = '0, w_word = '0;
    logic done;
    logic signed [31:0] acc_out;

    logic [31:0] a_mem[N*N_WORDS];
    logic [31:0] w_mem[N*N_WORDS];
    logic [31:0] acc_exp[N];
    int errors = 0;

    always #5 clk = ~clk;

    bnn_dot_top #(.N_BITS(N_BITS), .WORD_W(WORD_W)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .a_word(a_word), .w_word(w_word),
        .word_valid(word_valid), .last_word(last_word), .done(done), .acc_out(acc_out));

    initial begin
        $readmemh("vectors/bnn_a.hex", a_mem);
        $readmemh("vectors/bnn_w.hex", w_mem);
        $readmemh("vectors/bnn_acc.hex", acc_exp);
        repeat (2) @(posedge clk);
        rst_n <= 1;

        for (int t = 0; t < N; t++) begin
            for (int k = 0; k < N_WORDS; k++) begin
                @(posedge clk);
                start      <= (k == 0);
                a_word     <= a_mem[t*N_WORDS + k];
                w_word     <= w_mem[t*N_WORDS + k];
                word_valid <= 1;
                last_word  <= (k == N_WORDS - 1);
                if ($urandom_range(0, 3) == 0 && k != N_WORDS - 1) begin  // random stall
                    @(posedge clk);
                    start <= 0; word_valid <= 0; last_word <= 0;
                end
            end
            @(posedge clk);
            start <= 0; word_valid <= 0; last_word <= 0;
            @(posedge clk);  // done and acc_out are registered on the last-word edge
            if (acc_out !== $signed(acc_exp[t])) begin
                if (errors < 10)
                    $display("MISMATCH test %0d: got %0d expected %0d", t, acc_out,
                             $signed(acc_exp[t]));
                errors++;
            end
        end
        if (errors == 0) $display("PASS tb_bnn_dot: %0d dot products of %0d bits", N, N_BITS);
        else $display("FAIL tb_bnn_dot: %0d mismatches", errors);
        $finish;
    end
endmodule
