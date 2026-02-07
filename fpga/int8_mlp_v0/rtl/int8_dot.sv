module int8_dot #(
    parameter int IN = 8
)(
    input  logic signed [7:0] x   [IN],
    input  logic signed [7:0] w   [IN],
    output logic signed [31:0] acc
);
    integer i;
    logic signed [31:0] sum;
    logic signed [15:0] prod;

    always_comb begin
        sum = 0;
        for (i = 0; i < IN; i++) begin
            prod = $signed(x[i]) * $signed(w[i]);  // int8*int8 -> int16
            sum  = sum + $signed(prod);            // accumulate in int32
        end
        acc = sum;
    end
endmodule
