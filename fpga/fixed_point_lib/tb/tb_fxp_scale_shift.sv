module tb_fxp_scale_shift;
    localparam int NOUT=8;
    localparam int SHIFT=7;

    logic signed [31:0] in;
    logic signed [NOUT-1:0] out;
    fxp_scale_shift #(.NOUT(NOUT), .SHIFT(SHIFT)) dut(.in(in), .out(out));

    function int sat8(input int v);
        if (v > 127) return 127;
        if (v < -128) return -128;
        return v;
    endfunction

    function int rshift_round(input int v, input int s);
        if (s<=0) return v;
        return (v + (1<<(s-1))) >>> s;
    endfunction

    task check(input int vin);
        int q, exp;
        begin
            in = vin; #1;
            q = rshift_round(vin, SHIFT);
            exp = sat8(q);
            if ($signed(out) !== exp) begin
                $display("FAIL scale_shift: in=%0d got=%0d exp=%0d q=%0d", vin, $signed(out), exp, q);
                $finish;
            end
        end
    endtask

    initial begin
        check(0);
        check(128<<7);   // becomes 128 -> saturate to 127
        check(-200<<7);  // becomes -200 -> saturate to -128
        check(64<<7);    // becomes 64
        $display("PASS tb_fxp_scale_shift");
        $finish;
    end
endmodule
