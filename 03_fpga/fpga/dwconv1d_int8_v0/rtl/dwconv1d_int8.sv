module dwconv1d_int8 #(
    parameter int C = 4,
    parameter int L = 16,
    parameter int K = 3,
    parameter int SHIFT = 7
)(
    input  logic clk,
    input  logic rst,
    input  logic start,

    input  logic signed [7:0] x [C][L],
    output logic done,
    output logic signed [7:0] y [C][L-K+1]
);
    localparam int LOUT = L - K + 1;

    // Deterministic weights per channel
    logic signed [7:0] W [C][K];
    logic signed [31:0] B [C];

    // state
    int c_idx;
    int t_idx;
    int k_idx;

    logic busy;
    logic signed [31:0] acc;

    function automatic int sat8(input int v);
        if (v > 127) return 127;
        if (v < -128) return -128;
        return v;
    endfunction

    function automatic int rshift_round(input int v, input int s);
        if (s<=0) return v;
        return (v + (1<<(s-1))) >>> s;
    endfunction

    integer c, k;
    initial begin
        for (c = 0; c < C; c++) begin
            // simple bias pattern
            B[c] = (c - 1) <<< SHIFT;
            for (k = 0; k < K; k++) begin
                // weight pattern: e.g., [-1, 2, -1] scaled by (c+1)
                if (k == 0) W[c][k] = -1 * (c+1);
                else if (k == 1) W[c][k] =  2 * (c+1);
                else W[c][k] = -1 * (c+1);
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 0;
            done <= 0;
            c_idx <= 0;
            t_idx <= 0;
            k_idx <= 0;
            acc <= 0;
        end else begin
            done <= 0;

            if (!busy) begin
                if (start) begin
                    busy <= 1;
                    c_idx <= 0;
                    t_idx <= 0;
                    k_idx <= 0;
                    acc <= 0;
                end
            end else begin
                // accumulate over k for fixed (c_idx, t_idx)
                acc <= acc + $signed($signed(x[c_idx][t_idx + k_idx]) * $signed(W[c_idx][k_idx]));

                if (k_idx == K-1) begin
                    // finalize output for this (c,t)
                    int tmp;
                    tmp = $signed(acc) + $signed($signed(x[c_idx][t_idx + k_idx]) * $signed(W[c_idx][k_idx])) + $signed(B[c_idx]);
                    tmp = rshift_round(tmp, SHIFT);
                    y[c_idx][t_idx] <= sat8(tmp);

                    // move to next t/c
                    acc <= 0;
                    k_idx <= 0;

                    if (t_idx == LOUT-1) begin
                        t_idx <= 0;
                        if (c_idx == C-1) begin
                            busy <= 0;
                            done <= 1;
                        end else begin
                            c_idx <= c_idx + 1;
                        end
                    end else begin
                        t_idx <= t_idx + 1;
                    end
                end else begin
                    k_idx <= k_idx + 1;
                end
            end
        end
    end

endmodule
