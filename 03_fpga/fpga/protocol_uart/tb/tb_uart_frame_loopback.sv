module tb_uart_frame_loopback;

    localparam int MAXP = 255;

    logic clk=0, rst=1;
    always #5 clk = ~clk;

    // RX byte stream
    logic rx_valid;
    logic [7:0] rx_byte;

    // RX decoded frame
    logic frame_valid;
    logic frame_ready;
    logic [7:0] frame_len;
    logic [7:0] frame_type;
    logic [7:0] frame_payload [MAXP];
    logic crc_ok;

    uart_frame_rx #(.MAX_PAYLOAD(MAXP)) u_rx(
        .clk(clk), .rst(rst),
        .rx_valid(rx_valid), .rx_byte(rx_byte),
        .frame_valid(frame_valid), .frame_ready(frame_ready),
        .frame_len(frame_len), .frame_type(frame_type),
        .frame_payload(frame_payload), .crc_ok(crc_ok)
    );

    // TX uses RX frame
    logic tx_valid, tx_ready;
    logic [7:0] tx_byte;

    uart_frame_tx #(.MAX_PAYLOAD(MAXP)) u_tx(
        .clk(clk), .rst(rst),
        .frame_valid(frame_valid), .frame_ready(frame_ready),
        .frame_len(frame_len), .frame_type(frame_type),
        .frame_payload(frame_payload),
        .tx_valid(tx_valid), .tx_ready(tx_ready), .tx_byte(tx_byte)
    );

    // Ready always in TB
    initial begin
        tx_ready = 1;
        frame_ready = 1;
    end

    // Golden frame bytes (SOF LEN TYPE payload CRC)
    // We'll use LEN=8, TYPE=0x01, payload = [1..8]
    byte golden [0:11];

    function automatic [7:0] crc8_byte(input [7:0] c, input [7:0] d);
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
            crc8_byte = r;
        end
    endfunction

    function automatic [7:0] crc_frame(input [7:0] len, input [7:0] typ);
        integer i;
        reg [7:0] c;
        begin
            c = 8'h00;
            c = crc8_byte(c, len);
            c = crc8_byte(c, typ);
            for (i=0; i<len; i=i+1) c = crc8_byte(c, i+1);
            crc_frame = c;
        end
    endfunction

    initial begin
        // reset
        rx_valid = 0;
        rx_byte = 0;
        #20;
        rst = 0;

        golden[0] = 8'hA5;
        golden[1] = 8'd8;
        golden[2] = 8'h01;
        golden[3] = 8'd1;
        golden[4] = 8'd2;
        golden[5] = 8'd3;
        golden[6] = 8'd4;
        golden[7] = 8'd5;
        golden[8] = 8'd6;
        golden[9] = 8'd7;
        golden[10]= 8'd8;
        golden[11]= crc_frame(8, 8'h01);

        // feed bytes into RX
        for (int i=0; i<12; i++) begin
            @(posedge clk);
            rx_valid <= 1;
            rx_byte  <= golden[i];
        end
        @(posedge clk);
        rx_valid <= 0;

        // wait for decode
        repeat (5) @(posedge clk);

        if (!frame_valid) begin
            $display("FAIL: frame_valid never asserted");
            $finish;
        end
        if (!crc_ok) begin
            $display("FAIL: crc_ok=0");
            $finish;
        end
        if (frame_len !== 8 || frame_type !== 8'h01) begin
            $display("FAIL: decoded header wrong len=%0d type=%02h", frame_len, frame_type);
            $finish;
        end

        // Now watch TX output and compare
        int k = 0;
        while (k < 12) begin
            @(posedge clk);
            if (tx_valid) begin
                if (tx_byte !== golden[k]) begin
                    $display("FAIL: tx mismatch at %0d got=%02h exp=%02h", k, tx_byte, golden[k]);
                    $finish;
                end
                k++;
            end
        end

        $display("PASS tb_uart_frame_loopback");
        $finish;
    end

endmodule
