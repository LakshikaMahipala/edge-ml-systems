module tb_fxp_saturate;
    localparam int N=8;
    logic signed [31:0] in;
    logic signed [N-1:0] out;
    fxp_saturate #(.N(N)) dut(.in(in),.out(out));

    task check(input int vin, input int exp);
        begin
            in = vin; #1;
            if ($signed(out) !== exp) begin
                $display("FAIL sat: in=%0d got=%0d exp=%0d", vin, $signed(out), exp);
                $finish;
            end
        end
    endtask

    initial begin
        check(0, 0);
        check(200, 127);
        check(-300, -128);
        check(50, 50);
        $display("PASS tb_fxp_saturate");
        $finish;
    end
endmodule
