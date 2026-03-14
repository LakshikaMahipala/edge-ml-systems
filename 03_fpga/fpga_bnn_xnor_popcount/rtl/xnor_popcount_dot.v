module xnor_popcount_dot #(
    parameter integer N_BITS   = 256,  // total vector length
    parameter integer WORD_W   = 32,    // packing width
    parameter integer N_WORDS  = N_BITS / WORD_W
)(
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire                 start,
    input  wire [WORD_W-1:0]    a_word,
    input  wire [WORD_W-1:0]    w_word,
    input  wire                 word_valid,
    input  wire                 last_word,

    output reg                  done,
    output reg signed [31:0]    acc_out
);
    wire [WORD_W-1:0] xnor_bits = ~(a_word ^ w_word);

    // only WORD_W=32 supported here
    wire [5:0] pc;
    popcount32 u_pc(.x(xnor_bits[31:0]), .count(pc));

    // signed contribution: 2*pc - WORD_W
    wire signed [31:0] contrib = $signed({1'b0, pc}) * 2 - WORD_W;

    reg signed [31:0] acc;

    always @(posedge clk) begin
        if (!rst_n) begin
            acc     <= 0;
            done    <= 0;
            acc_out <= 0;
        end else begin
            done <= 0;

            if (start) begin
                acc <= 0;
            end

            if (word_valid) begin
                acc <= acc + contrib;
                if (last_word) begin
                    acc_out <= acc + contrib;
                    done <= 1'b1;
                end
            end
        end
    end

endmodule
