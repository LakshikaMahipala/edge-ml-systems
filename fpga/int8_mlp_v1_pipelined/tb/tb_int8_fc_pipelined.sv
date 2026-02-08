module tb_int8_fc_pipelined;

    localparam int IN=8;
    localparam int OUT=4;
    localparam int SHIFT=7;

    logic clk=0, rst=1;
    always #5 clk = ~clk;

    logic start, done;
    logic signed [7:0] x [IN];
    logic signed [7:0] y [OUT];

    int8_fc_pipelined #(.IN(IN), .OUT(OUT), .SHIFT(SHIFT)) dut(
        .clk(clk), .rst(rst),
        .start(start),
        .x(x),
        .done(done),
        .y(y)
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

    task compute_expected(output int exp [OUT]);
        int oi, ii;
        int acc;
        int w;
        int b;
        int prod;
        begin
            for (oi=0; oi<OUT; oi++) begin
                acc = 0;
                for (ii=0; ii<IN; ii++) begin
                    w = (oi+1) * (ii-3);
                    prod = $signed(x[ii]) * w;
                    acc += prod;
                end
                if (oi==0) b =  10 <<< SHIFT;
                else if (oi==1) b = -20 <<< SHIFT;
                else if (oi==2) b =   5 <<< SHIFT;
                else b = 0;
                acc += b;
                exp[oi] = sat8(rshift_round(acc, SHIFT));
            end
        end
    endtask

    initial begin
        int exp [OUT];

        start = 0;
        x[0]= 10; x[1]= -3; x[2]= 7; x[3]= 2;
        x[4]= -8; x[5]= 1;  x[6]= 4; x[7]= -2;

        #20; rst=0;

        compute_expected(exp);

        @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;

        // wait for done
        wait(done == 1);
        @(posedge clk);

        for (int j=0; j<OUT; j++) begin
            if ($signed(y[j]) !== exp[j]) begin
                $display("FAIL j=%0d got=%0d exp=%0d", j, $signed(y[j]), exp[j]);
                $finish;
            end
        end

        $display("PASS tb_int8_fc_pipelined");
        $finish;
    end

endmodule
