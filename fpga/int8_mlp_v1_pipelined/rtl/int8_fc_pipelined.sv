module int8_fc_pipelined #(
    parameter int IN = 8,
    parameter int OUT = 4,
    parameter int SHIFT = 7
)(
    input  logic clk,
    input  logic rst,

    input  logic start,
    input  logic signed [7:0] x [IN],

    output logic done,
    output logic signed [7:0] y [OUT]
);
    // weights/bias constants (same deterministic pattern as v0)
    logic signed [7:0]  W [OUT][IN];
    logic signed [31:0] B [OUT];

    // accumulators for OUT neurons
    logic signed [31:0] acc [OUT];

    // input index counter
    logic [$clog2(IN+1)-1:0] idx;
    logic busy;

    function automatic int sat8(input int v);
        if (v > 127) return 127;
        if (v < -128) return -128;
        return v;
    endfunction

    function automatic int rshift_round(input int v, input int s);
        if (s<=0) return v;
        return (v + (1<<(s-1))) >>> s;
    endfunction

    integer oi, ii;
    initial begin
        B[0] =  10 <<< SHIFT;
        B[1] = -20 <<< SHIFT;
        B[2] =   5 <<< SHIFT;
        B[3] =   0 <<< SHIFT;

        for (oi=0; oi<OUT; oi++) begin
            for (ii=0; ii<IN; ii++) begin
                W[oi][ii] = (oi+1) * (ii-3);
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 0;
            idx <= 0;
            done <= 0;
            for (int j=0; j<OUT; j++) acc[j] <= 0;
        end else begin
            done <= 0;

            if (!busy) begin
                if (start) begin
                    busy <= 1;
                    idx <= 0;
                    for (int j=0; j<OUT; j++) acc[j] <= 0;
                end
            end else begin
                // one IN element per cycle; update all OUT accumulators in parallel
                for (int j=0; j<OUT; j++) begin
                    acc[j] <= acc[j] + $signed($signed(x[idx]) * $signed(W[j][idx]));
                end

                if (idx == IN-1) begin
                    busy <= 0;
                    done <= 1;

                    // finalize y with bias + requant + saturate
                    for (int j=0; j<OUT; j++) begin
                        int tmp;
                        tmp = $signed(acc[j]) + $signed($signed(x[idx]) * $signed(W[j][idx])) + $signed(B[j]);
                        tmp = rshift_round(tmp, SHIFT);
                        y[j] <= sat8(tmp);
                    end
                end else begin
                    idx <= idx + 1;
                end
            end
        end
    end
endmodule
