module tb_fxp_mul;
    localparam int N=8;
    localparam int F=7;

    logic signed [N-1:0] a,b,y;
    fxp_mul #(.N(N),.F(F)) dut(.a(a),.b(b),.y(y));

    function int sat8(input int v);
        if (v > 127) return 127;
        if (v < -128) return -128;
        return v;
    endfunction

    function int rshift_round(input int v, input int s);
        if (s<=0) return v;
        return (v + (1<<(s-1))) >>> s;
    endfunction

    task check(input int ai, input int bi);
        int prod, q, exp;
        begin
            a = ai; b = bi; #1;
            prod = ai * bi;
            q = rshift_round(prod, F);
            exp = sat8(q);
            if ($signed(y) !== exp) begin
                $display("FAIL mul: a=%0d b=%0d got=%0d exp=%0d (prod=%0d q=%0d)", ai, bi, $signed(y), exp, prod, q);
                $finish;
            end
        end
    endtask

    initial begin
        // ~0.5 in Q1.7 is 64
        check(64, 64);     // 0.5*0.5=0.25 => 32
        check(127, 127);   // near 1.0*1.0 => saturates near 127
        check(-128, 127);  // -1 * ~1 => -128 or -127 depending on rounding/sat
        check(50, -50);
        $display("PASS tb_fxp_mul");
        $finish;
    end
endmodule
