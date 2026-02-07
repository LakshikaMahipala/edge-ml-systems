module uart_frame_rx #(
    parameter int MAX_PAYLOAD = 255
)(
    input  logic clk,
    input  logic rst,

    // Byte stream input from UART RX
    input  logic       rx_valid,
    input  logic [7:0] rx_byte,

    // Decoded frame output
    output logic       frame_valid,
    input  logic       frame_ready,
    output logic [7:0] frame_len,
    output logic [7:0] frame_type,
    output logic [7:0] frame_payload [MAX_PAYLOAD],
    output logic       crc_ok
);
    typedef enum logic [2:0] {WAIT_SOF, GET_LEN, GET_TYPE, GET_PAYLOAD, GET_CRC, HOLD} state_t;
    state_t st;

    logic [7:0] idx;
    logic [7:0] crc_calc;

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

    always_ff @(posedge clk) begin
        if (rst) begin
            st <= WAIT_SOF;
            frame_len <= 0;
            frame_type <= 0;
            idx <= 0;
            crc_calc <= 0;
            crc_ok <= 0;
            frame_valid <= 0;
        end else begin
            if (st == HOLD) begin
                if (frame_ready) begin
                    frame_valid <= 0;
                    st <= WAIT_SOF;
                end
            end else if (rx_valid) begin
                case (st)
                    WAIT_SOF: begin
                        if (rx_byte == 8'hA5) begin
                            st <= GET_LEN;
                            crc_calc <= 8'h00;
                        end
                    end
                    GET_LEN: begin
                        frame_len <= rx_byte;
                        crc_calc <= crc8_byte(8'h00, rx_byte);
                        st <= GET_TYPE;
                    end
                    GET_TYPE: begin
                        frame_type <= rx_byte;
                        crc_calc <= crc8_byte(crc_calc, rx_byte);
                        idx <= 0;
                        st <= (frame_len == 0) ? GET_CRC : GET_PAYLOAD;
                    end
                    GET_PAYLOAD: begin
                        frame_payload[idx] <= rx_byte;
                        crc_calc <= crc8_byte(crc_calc, rx_byte);
                        idx <= idx + 1;
                        if (idx + 1 >= frame_len) st <= GET_CRC;
                    end
                    GET_CRC: begin
                        crc_ok <= (rx_byte == crc_calc);
                        frame_valid <= 1;
                        st <= HOLD;
                    end
                    default: st <= WAIT_SOF;
                endcase
            end
        end
    end
endmodule
