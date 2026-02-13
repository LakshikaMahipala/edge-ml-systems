`timescale 1ns/1ps

module tb_bnn_dot_top;
    localparam integer N_BITS = 64;
    localparam integer WORD_W = 32;

    reg clk = 0;
    always #5 clk = ~clk;

    reg rst_n = 0;

    reg start;
    reg [WORD_W-1:0] a_word;
    reg [WORD_W-1:0] w_word;
    reg word_valid;
    reg last_word;

    wire done;
    wire signed [31:0] acc_out;

    bnn_dot_top #(.N_BITS(N_BITS), .WORD_W(WORD_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .a_word(a_word),
        .w_word(w_word),
        .word_valid(word_valid),
        .last_word(last_word),
        .done(done),
        .acc_out(acc_out)
    );

    // helper to compute expected signed sum:
    // contrib = 2*popcount(xnor) - 32, summed over words.

    initial begin
        start = 0; a_word = 0; w_word = 0; word_valid = 0; last_word = 0;

        #20;
        rst_n = 1;

        // Example:
        // word0: a=all1, w=all1 => xnor=all1 popcount=32 => contrib=64-32=+32
        // word1: a=all1, w=all0 => xnor=all0 popcount=0  => contrib=0-32=-32
        // total expected = 0
        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        // send word0
        @(posedge clk);
        a_word <= 32'hFFFFFFFF;
        w_word <= 32'hFFFFFFFF;
        word_valid <= 1;
        last_word <= 0;

        // send word1 (last)
        @(posedge clk);
        a_word <= 32'hFFFFFFFF;
        w_word <= 32'h00000000;
        word_valid <= 1;
        last_word <= 1;

        // stop valid
        @(posedge clk);
        word_valid <= 0;
        last_word <= 0;

        // wait for done
        wait(done == 1'b1);
        $display("ACC_OUT = %0d", acc_out);

        if (acc_out !== 0) begin
            $display("FAIL: expected 0");
            $finish(1);
        end else begin
            $display("PASS");
        end

        #20;
        $finish(0);
    end
endmodule
