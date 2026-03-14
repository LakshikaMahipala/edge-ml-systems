module crc8 (
    input  logic       clk,
    input  logic       rst,
    input  logic       init,
    input  logic       en,
    input  logic [7:0] data,
    output logic [7:0] crc
);
    // CRC-8 poly: x^8 + x^2 + x + 1 (0x07)
    function automatic [7:0] next_crc(input [7:0] c, input [7:0] d);
        integer i;
        reg [7:0] r;
        reg fb;
        begin
            r = c;
            for (i=0; i<8; i=i+1) begin
                fb = r[7] ^ d[7-i];
                r  = {r[6:0], 1'b0};
                if (fb) r = r ^ 8'h07;
            end
            next_crc = r;
        end
    endfunction

    always_ff @(posedge clk) begin
        if (rst) crc <= 8'h00;
        else if (init) crc <= 8'h00;
        else if (en) crc <= next_crc(crc, data);
    end
endmodule
