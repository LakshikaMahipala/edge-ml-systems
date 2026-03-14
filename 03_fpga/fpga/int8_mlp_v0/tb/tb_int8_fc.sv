module tb_int8_fc;

    localparam int IN=8;
    localparam int OUT=4;
    localparam int SHIFT=7;

    logic signed [7:0] x [IN];
    logic signed [7:0] y [OUT];

    int8_fc #(.IN(IN), .OUT(OUT), .SHIFT(SHIFT)) dut(.x(x), .y(y));

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
                    w = (oi+1) * (ii-3);     // must match RTL
                    prod = $signed(x[ii]) * w;
                    acc += prod;
                end
                // bias
                if (oi==0) b =  10 <<< SHIFT;
                else if (oi==1) b = -20 <<< SHIFT;
                else if (oi==2) b =   5 <<< SHIFT;
                else b = 0;
                acc += b;

                // requant + sat
                exp[oi] = sat8(rshift_round(acc, SHIFT));
            end
        end
    endtask

    initial begin
        int exp [OUT];

        // deterministic input vector
        x[0]= 10; x[1]= -3; x[2]= 7; x[3]= 2;
        x[4]= -8; x[5]= 1;  x[6]= 4; x[7]= -2;

        #1;
        compute_expected(exp);

        for (int j=0; j<OUT; j++) begin
            if ($signed(y[j]) !== exp[j]) begin
                $display("FAIL j=%0d got=%0d exp=%0d", j, $signed(y[j]), exp[j]);
                $finish;
            end
        end

        $display("PASS tb_int8_fc");
        $finish;
    end

endmodule
