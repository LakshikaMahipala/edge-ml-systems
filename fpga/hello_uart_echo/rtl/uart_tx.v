module uart_tx #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 115200
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [7:0] data,
    output reg  tx,
    output reg  busy
);
    localparam int CLKS_PER_BIT = CLK_HZ / BAUD;

    typedef enum logic [2:0] {IDLE, START, DATA, STOP} state_t;
    state_t state;

    reg [$clog2(CLKS_PER_BIT+1)-1:0] clk_cnt;
    reg [2:0] bit_idx;
    reg [7:0] sh;

    always @(posedge clk) begin
        if (!rst_n) begin
            state   <= IDLE;
            tx      <= 1'b1;
            busy    <= 1'b0;
            clk_cnt <= 0;
            bit_idx <= 0;
            sh      <= 0;
        end else begin
            case (state)
                IDLE: begin
                    tx      <= 1'b1;
                    busy    <= 1'b0;
                    clk_cnt <= 0;
                    bit_idx <= 0;
                    if (start) begin
                        sh   <= data;
                        busy <= 1'b1;
                        state <= START;
                    end
                end

                START: begin
                    tx <= 1'b0;
                    if (clk_cnt == CLKS_PER_BIT-1) begin
                        clk_cnt <= 0;
                        state   <= DATA;
                    end else clk_cnt <= clk_cnt + 1;
                end

                DATA: begin
                    tx <= sh[bit_idx];
                    if (clk_cnt == CLKS_PER_BIT-1) begin
                        clk_cnt <= 0;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 0;
                            state   <= STOP;
                        end else bit_idx <= bit_idx + 1;
                    end else clk_cnt <= clk_cnt + 1;
                end

                STOP: begin
                    tx <= 1'b1;
                    if (clk_cnt == CLKS_PER_BIT-1) begin
                        clk_cnt <= 0;
                        state   <= IDLE;
                    end else clk_cnt <= clk_cnt + 1;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
