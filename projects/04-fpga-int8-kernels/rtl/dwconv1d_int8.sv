// INT8 depthwise 1D convolution (valid padding, stride 1), one MAC per cycle.
//
//     y[c][t] = requant_int8( b[c] + sum_k x[c][t+k] * W[c][k] , SHIFT )
//
// Depthwise = each channel has its own K-tap filter and channels never mix.
// It is the cheap half of a depthwise-separable block (MobileNet-style); the
// pointwise half is a 1x1 convolution, i.e. int8_fc applied at every t.
//
// Data layout (packed buses, element e = bits [8*e +: 8]):
//   x_flat: x[c][t] at e = c*L + t           y_flat: y[c][t] at e = c*LOUT + t
//   W_FILE: W[c][k] at line c*K + k (int8)   B_FILE: b[c] at line c (int32)
//
// Timing: C * LOUT * (K + 1) cycles from `start` to `done`
//   (K MAC cycles + 1 write cycle per output).
module dwconv1d_int8 #(
    parameter int C      = 4,
    parameter int L      = 32,
    parameter int K      = 3,
    parameter int SHIFT  = 6,
    parameter     W_FILE = "vectors/dw_weights.hex",
    parameter     B_FILE = "vectors/dw_bias.hex"
) (
    input  logic                clk,
    input  logic                rst,
    input  logic                start,
    input  logic [C*L*8-1:0]    x_flat,
    output logic                busy,
    output logic                done,
    output logic [C*(L-K+1)*8-1:0] y_flat
);
    import fxp_pkg::*;
    localparam int LOUT = L - K + 1;

    logic signed [7:0]  w_rom[C*K];
    logic signed [31:0] b_rom[C];
    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    typedef enum logic [1:0] {IDLE, MAC, WRITE} state_t;
    state_t state;

    int c, t, k;                 // channel, output position, filter tap
    logic signed [31:0] acc;
    logic signed [7:0]  x_cur;

    assign x_cur = x_flat[8*(c*L + t + k) +: 8];
    assign busy  = (state != IDLE);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done  <= 1'b0;
            c <= 0; t <= 0; k <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    c <= 0; t <= 0; k <= 0;
                    acc   <= b_rom[0];
                    state <= MAC;
                end
                MAC: begin
                    acc <= acc + x_cur * w_rom[c*K + k];
                    if (k == K - 1) state <= WRITE;
                    else k <= k + 1;
                end
                WRITE: begin
                    y_flat[8*(c*LOUT + t) +: 8] <= requant_int8(acc, SHIFT);
                    k <= 0;
                    if (t == LOUT - 1) begin
                        t <= 0;
                        if (c == C - 1) begin
                            done  <= 1'b1;
                            state <= IDLE;
                        end else begin
                            c     <= c + 1;
                            acc   <= b_rom[c + 1];
                            state <= MAC;
                        end
                    end else begin
                        t     <= t + 1;
                        acc   <= b_rom[c];
                        state <= MAC;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
