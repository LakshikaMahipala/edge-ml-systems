module tb_dwconv1d_int8;

    localparam int C=4;
    localparam int L=16;
    localparam int K=3;
    localparam int SHIFT=7;
    localparam int LOUT = L-K+1;

    logic clk=0, rst=1, start;
    always #5 clk = ~clk;

    logic signed [7:0] x [C][L];
    logic signed [7:0] y [C][LOUT];
    logic done;

    dwconv1d_int8 #(.C(C), .L(L), .K(K), .SHIFT(SHIFT)) dut(
        .clk(clk), .rst(rst), .start(start),
        .x(x), .done(done), .y(y)
    );

    function int sat8(input int v);
        if (v > 127) return 127;
        if (v < -128) return -128;
        return v;
    endfunction

    function int rshift_round(input int v, input int s);
        if (s<=0) return v;
        return (v + (1<<(s-1))) >>> s;
    endfunction

    task build_expected(output int exp [C][LOUT]);
        int c,t,k;
        int acc;
        int w;
        int b;
        begin
            for (c=0; c<C; c++) begin
                b = (c - 1) <<< SHIFT;
                for (t=0; t<LOUT; t++) begin
                    acc = 0;
                    for (k=0; k<K; k++) begin
                        if (k==0) w = -1 * (c+1);
                        else if (k==1) w = 2 * (c+1);
                        else w = -1 * (c+1);
                        acc += $signed(x[c][t+k]) * w;
                    end
                    acc += b;
                    exp[c][t] = sat8(rshift_round(acc, SHIFT));
                end
            end
        end
    endtask

    initial begin
        int exp [C][LOUT];

        start = 0;

        // deterministic input pattern
        for (int c=0; c<C; c++) begin
            for (int i=0; i<L; i++) begin
                x[c][i] = $signed((c*7 + i*3) % 31) - 15;
            end
        end

        #20; rst=0;

        build_expected(exp);

        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        wait(done == 1);
        @(posedge clk);

        for (int c=0; c<C; c++) begin
            for (int t=0; t<LOUT; t++) begin
                if ($signed(y[c][t]) !== exp[c][t]) begin
                    $display("FAIL c=%0d t=%0d got=%0d exp=%0d", c, t, $signed(y[c][t]), exp[c][t]);
                    $finish;
                end
            end
        end

        $display("PASS tb_dwconv1d_int8");
        $finish;
    end

endmodule
