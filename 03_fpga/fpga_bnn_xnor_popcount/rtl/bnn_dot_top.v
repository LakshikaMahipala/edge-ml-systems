module bnn_dot_top #(
    parameter integer N_BITS  = 256,
    parameter integer WORD_W  = 32,
    parameter integer N_WORDS = N_BITS / WORD_W
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 start,
    input  wire [WORD_W-1:0]    a_word,
    input  wire [WORD_W-1:0]    w_word,
    input  wire                 word_valid,
    input  wire                 last_word,

    output wire                 done,
    output wire signed [31:0]   acc_out
);
    xnor_popcount_dot #(
        .N_BITS(N_BITS),
        .WORD_W(WORD_W),
        .N_WORDS(N_WORDS)
    ) u_dot (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .a_word(a_word),
        .w_word(w_word),
        .word_valid(word_valid),
        .last_word(last_word),
        .done(done),
        .acc_out(acc_out)
    );
endmodule
