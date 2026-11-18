// INT8 fully-connected layer, fully parallel (combinational).
//
//     y[j] = requant_int8( b[j] + sum_i x[i] * W[j][i] , SHIFT )
//
// - x, W: int8      - b, accumulator: int32      - y: int8
// - OUT x IN multipliers: one output vector per clock if registered outside.
//   Best for small layers; see int8_fc_seq.sv for the resource-saving version.
// - Weights/biases are ROMs initialised from hex files produced by
//   golden/gen_vectors.py (Vivado and Quartus infer ROMs from $readmemh).
// - Vectors are flattened into packed buses (element i = bits [8*i +: 8]) for
//   portability across simulators and synthesis tools.
module int8_fc #(
    parameter int IN     = 16,
    parameter int OUT    = 8,
    parameter int SHIFT  = 8,
    parameter     W_FILE = "vectors/fc_weights.hex",  // OUT*IN int8, row-major W[j][i]
    parameter     B_FILE = "vectors/fc_bias.hex"      // OUT int32
) (
    input  logic [IN*8-1:0]  x_flat,
    output logic [OUT*8-1:0] y_flat
);
    import fxp_pkg::*;

    logic signed [7:0]  w_rom[OUT*IN];
    logic signed [31:0] b_rom[OUT];
    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    logic signed [31:0] acc;
    always_comb begin
        y_flat = '0;
        for (int j = 0; j < OUT; j++) begin
            acc = b_rom[j];
            for (int i = 0; i < IN; i++)
                acc = acc + $signed(x_flat[8*i +: 8]) * w_rom[j*IN + i];
            y_flat[8*j +: 8] = requant_int8(acc, SHIFT);
        end
    end
endmodule
