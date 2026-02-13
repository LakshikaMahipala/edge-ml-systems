module popcount32 (
    input  wire [31:0] x,
    output wire [5:0]  count
);
    wire [1:0] s0 [15:0];
    genvar i;
    generate
        for (i=0; i<16; i=i+1) begin : GEN_PAIR
            assign s0[i] = x[2*i] + x[2*i+1];
        end
    endgenerate

    wire [2:0] s1 [7:0];
    generate
        for (i=0; i<8; i=i+1) begin : GEN_QUAD
            assign s1[i] = s0[2*i] + s0[2*i+1];
        end
    endgenerate

    wire [3:0] s2 [3:0];
    generate
        for (i=0; i<4; i=i+1) begin : GEN_OCT
            assign s2[i] = s1[2*i] + s1[2*i+1];
        end
    endgenerate

    wire [4:0] s3 [1:0];
    assign s3[0] = s2[0] + s2[1];
    assign s3[1] = s2[2] + s2[3];

    assign count = s3[0] + s3[1]; // max 32
endmodule
