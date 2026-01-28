module tb_uart_echo;

    localparam int CLK_HZ = 10_000_000;   // use smaller clock for sim speed
    localparam int BAUD   = 115200;

    logic clk = 0;
    logic rst_n = 0;

    logic rx;
    wire  tx;

    // DUT
    uart_echo_top #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx_i(rx),
        .uart_tx_o(tx)
    );

    // Clock
    always #50 clk = ~clk; // 10MHz => period 100ns

    // UART bit time
    real bit_time_ns;
    initial bit_time_ns = 1e9 / BAUD;

    task send_uart_byte(input byte b);
        int i;
        begin
            // start bit
            rx = 0;
            #(bit_time_ns);
            // data bits LSB first
            for (i = 0; i < 8; i++) begin
                rx = b[i];
                #(bit_time_ns);
            end
            // stop bit
            rx = 1;
            #(bit_time_ns);
        end
    endtask

    // Simple sampler for TX to reconstruct a byte
    task recv_uart_byte(output byte b);
        int i;
        begin
            // wait for start bit (tx goes low)
            wait (tx == 0);
            // sample mid-bit of start
            #(bit_time_ns/2);
            // now sample each data bit at 1-bit intervals
            #(bit_time_ns);
            for (i = 0; i < 8; i++) begin
                b[i] = tx;
                #(bit_time_ns);
            end
            // stop bit
            #(bit_time_ns);
        end
    endtask

    byte sent [0:2];
    byte got;

    initial begin
        rx = 1;
        sent[0] = 8'h3A;
        sent[1] = 8'hA5;
        sent[2] = 8'h7E;

        // reset
        #500;
        rst_n = 1;

        // send and expect echo
        foreach (sent[idx]) begin
            send_uart_byte(sent[idx]);
            recv_uart_byte(got);

            if (got !== sent[idx]) begin
                $display("FAIL idx=%0d expected=%02x got=%02x", idx, sent[idx], got);
                $finish;
            end else begin
                $display("PASS idx=%0d byte=%02x", idx, got);
            end
        end

        $display("ALL PASS");
        $finish;
    end

endmodule
