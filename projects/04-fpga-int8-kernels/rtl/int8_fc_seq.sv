// INT8 fully-connected layer, sequential: OUT parallel MACs, one input per cycle.
//
// Same maths as int8_fc.sv, but uses only OUT multipliers instead of OUT*IN:
// every cycle, input element x[idx] is multiplied by column idx of W and
// added to all OUT accumulators.
//
// Timing (cycles from the `start` pulse to `done`): IN + 1
//   - IN cycles of multiply-accumulate (accumulators are preloaded with bias)
//   - 1 cycle to requantize and register the outputs
// Throughput: one vector every IN + 1 cycles (no double buffering).
//
// Handshake: pulse `start` for one cycle while `busy` is low; x_flat must stay
// stable until `done`. `done` pulses high for one cycle when y_flat is valid.
module int8_fc_seq #(
    parameter int IN     = 16,
    parameter int OUT    = 8,
    parameter int SHIFT  = 8,
    parameter     W_FILE = "vectors/fc_weights.hex",
    parameter     B_FILE = "vectors/fc_bias.hex"
) (
    input  logic             clk,
    input  logic             rst,     // synchronous, active high
    input  logic             start,
    input  logic [IN*8-1:0]  x_flat,
    output logic             busy,
    output logic             done,
    output logic [OUT*8-1:0] y_flat
);
    import fxp_pkg::*;

    logic signed [7:0]  w_rom[OUT*IN];
    logic signed [31:0] b_rom[OUT];
    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    typedef enum logic [1:0] {IDLE, MAC, FINISH} state_t;
    state_t state;

    logic signed [31:0] acc[OUT];
    logic [$clog2(IN)-1:0] idx;
    logic signed [7:0] x_cur;

    assign x_cur = x_flat[8*idx +: 8];
    assign busy  = (state != IDLE);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            idx   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    for (int j = 0; j < OUT; j++) acc[j] <= b_rom[j];
                    idx   <= '0;
                    state <= MAC;
                end
                MAC: begin
                    for (int j = 0; j < OUT; j++)
                        acc[j] <= acc[j] + x_cur * w_rom[j*IN + idx];
                    if (idx == IN - 1) state <= FINISH;
                    else idx <= idx + 1'b1;
                end
                FINISH: begin
                    for (int j = 0; j < OUT; j++) y_flat[8*j +: 8] <= requant_int8(acc[j], SHIFT);
                    done  <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
