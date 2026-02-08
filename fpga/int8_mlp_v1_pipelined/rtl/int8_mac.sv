module int8_mac(
    input  logic signed [7:0]  x,
    input  logic signed [7:0]  w,
    input  logic signed [31:0] acc_in,
    output logic signed [31:0] acc_out
);
    logic signed [15:0] prod;
    always_comb begin
        prod = $signed(x) * $signed(w);
        acc_out = acc_in + $signed(prod);
    end
endmodule
