module uart_echo_top #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 115200
)(
    input  wire clk,
    input  wire rst_n,
    input  wire uart_rx_i,
    output wire uart_tx_o
);
    wire [7:0] rx_data;
    wire       rx_valid;

    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy;

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst_n(rst_n),
        .rx(uart_rx_i),
        .data(rx_data),
        .valid(rx_valid)
    );

    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst_n(rst_n),
        .start(tx_start),
        .data(tx_data),
        .tx(uart_tx_o),
        .busy(tx_busy)
    );

    // Echo logic: when byte arrives, transmit it if TX not busy
    always @(posedge clk) begin
        if (!rst_n) begin
            tx_start <= 1'b0;
            tx_data  <= 8'h00;
        end else begin
            tx_start <= 1'b0;
            if (rx_valid && !tx_busy) begin
                tx_data  <= rx_data;
                tx_start <= 1'b1;
            end
        end
    end
endmodule
