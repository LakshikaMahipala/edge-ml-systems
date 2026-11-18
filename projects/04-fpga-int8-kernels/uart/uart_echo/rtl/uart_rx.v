module uart_rx #(
    parameter int CLK_HZ = 50_000_000,
    parameter int BAUD   = 115200
)(
    input  wire clk,
    input  wire rst_n,
    input  wire rx,
    output reg  [7:0] data,
    output reg  valid
);
    localparam int CLKS_PER_BIT = CLK_HZ / BAUD;

    typedef enum logic [2:0] {IDLE, START, DATA, STOP, DONE} state_t;
    state_t state;

    reg [$clog2(CLKS_PER_BIT+1)-1:0] clk_cnt;
    reg [2:0] bit_idx;
    reg [7:0] shift;

    always @(posedge clk) begin
        if (!rst_n) begin
            state   <= IDLE;
            clk_cnt <= 0;
            bit_idx <= 0;
            shift   <= 0;
            data    <= 0;
            valid   <= 0;
        end else begin
            valid <= 0;
            case (state)
                IDLE: begin
                    clk_cnt <= 0;
                    bit_idx <= 0;
                    if (rx == 1'b0) state <= START; // start bit
                end

                START: begin
                    // sample mid start-bit
                    if (clk_cnt == (CLKS_PER_BIT/2)) begin
                        if (rx == 1'b0) begin
                            clk_cnt <= 0;
                            state   <= DATA;
                        end else begin
                            state <= IDLE; // false start
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                DATA: begin
                    if (clk_cnt == CLKS_PER_BIT-1) begin
                        clk_cnt <= 0;
                        shift[bit_idx] <= rx;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 0;
                            state   <= STOP;
                        end else begin
                            bit_idx <= bit_idx + 1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                STOP: begin
                    if (clk_cnt == CLKS_PER_BIT-1) begin
                        clk_cnt <= 0;
                        data  <= shift;
                        valid <= 1;
                        state <= DONE;
                    end else begin
                        clk_cnt <= clk_cnt + 1;
                    end
                end

                DONE: begin
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end
endmodule
