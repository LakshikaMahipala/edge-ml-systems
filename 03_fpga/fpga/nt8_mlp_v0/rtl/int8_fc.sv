module int8_fc #(
    parameter int IN = 8,
    parameter int OUT = 4,
    parameter int SHIFT = 7
)(
    input  logic signed [7:0] x [IN],
    output logic signed [7:0] y [OUT]
);
    // Hardcoded weights (OUT x IN) and bias (OUT)
    // These values are arbitrary but fixed so TB can check exactly.
    logic signed [7:0] W [OUT][IN];
    logic signed [31:0] B [OUT];

    // Accumulators
    logic signed [31:0] acc [OUT];

    genvar j;
    generate
        for (j = 0; j < OUT; j++) begin : GEN_OUT
            int8_dot #(.IN(IN)) u_dot (
                .x(x),
                .w(W[j]),
                .acc(acc[j])
            );

            // bias add + requant
            logic signed [31:0] acc_b;
            always_comb acc_b = acc[j] + B[j];

            int8_requant #(.SHIFT(SHIFT)) u_rq (
                .acc(acc_b),
                .y(y[j])
            );
        end
    endgenerate

    // Initialize constants (synthesizable as ROM constants)
    integer oi, ii;
    initial begin
        // Biases
        B[0] =  10 <<< SHIFT;
        B[1] = -20 <<< SHIFT;
        B[2] =   5 <<< SHIFT;
        B[3] =   0 <<< SHIFT;

        // Weights (small integers)
        for (oi=0; oi<OUT; oi++) begin
            for (ii=0; ii<IN; ii++) begin
                W[oi][ii] = (oi+1) * (ii-3); // deterministic pattern
            end
        end
    end

endmodule
