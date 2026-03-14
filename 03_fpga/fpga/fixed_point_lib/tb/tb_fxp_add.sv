module tb_fxp_add;
    localparam int N=8;
    logic signed [N-1:0] a,b,y;
    fxp_add #(.N(N)) dut(.a(a),.b(b),.y(y));

    task check(input int ai, input int bi);
        int exp;
        begin
            a = ai; b = bi; #1;
            exp = ai + bi;
            if (exp > 127) exp = 127;
            if (exp < -128) exp = -128;
            if ($signed(y) !== exp) begin
                $display("FAIL add: a=%0d b=%0d got=%0d exp=%0d", ai, bi, $signed(y), exp);
                $finish;
            end
        end
    endtask

    initial begin
        check(10, 20);
        check(100, 60);     // saturate
        check(-100, -60);   // saturate
        check(127, 1);
        check(-128, -1);
        $display("PASS tb_fxp_add");
        $finish;
    end
endmodule
