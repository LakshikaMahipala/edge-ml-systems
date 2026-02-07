module uart_frame_tx #(
    parameter int MAX_PAYLOAD = 255
)(
    input  logic clk,
    input  logic rst,

    // Frame input
    input  logic       frame_valid,
    output logic       frame_ready,
    input  logic [7:0] frame_len,
    input  logic [7:0] frame_type,
    input  logic [7:0] frame_payload [MAX_PAYLOAD],

    // Byte stream output to UART TX
    output logic       tx_valid,
    input  logic       tx_ready,
    output logic [7:0] tx_byte
);
    typedef enum logic [2:0] {IDLE, SOF, LEN, TYPE, PAYLOAD, CRC} state_t;
    state_t st;

    logic [7:0] idx;
    logic [7:0] crc_val;

    // Simple combinational CRC for v0 (since payload small). For MAX payload, you'd stream CRC.
    function automatic [7:0] crc8_calc(input [7:0] len, input [7:0] typ, input logic [7:0] payload [MAX_PAYLOAD]);
        integer i;
        reg [7:0] c;
        reg [7:0] b;
        begin
            c = 8'h00;
            // local helper: crc update (same poly as crc8.sv)
            for (i=0; i<1; i=i+1) begin end
            // update with LEN
            b = len; c = crc8_byte(c,b);
            // update with TYPE
            b = typ; c = crc8_byte(c,b);
            // payload
            for (i=0; i<len; i=i+1) begin
                b = payload[i];
                c = crc8_byte(c,b);
            end
            crc8_calc = c;
        end
    endfunction

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
            st <= IDLE;
            idx <= 0;
            crc_val <= 0;
        end else begin
            if (st == IDLE) begin
                if (frame_valid) begin
                    idx <= 0;
                    crc_val <= crc8_calc(frame_len, frame_type, frame_payload);
                    st <= SOF;
                end
            end else if (tx_ready) begin
                case (st)
                    SOF:     st <= LEN;
                    LEN:     st <= TYPE;
                    TYPE:    st <= (frame_len == 0) ? CRC : PAYLOAD;
                    PAYLOAD: st <= (idx+1 >= frame_len) ? CRC : PAYLOAD;
                    CRC:     st <= IDLE;
                    default: st <= IDLE;
                endcase

                if (st == PAYLOAD) idx <= idx + 1;
                else idx <= 0;
            end
        end
    end

    always_comb begin
        frame_ready = (st == IDLE);

        tx_valid = (st != IDLE);
        tx_byte  = 8'h00;

        case (st)
            SOF:     tx_byte = 8'hA5;
            LEN:     tx_byte = frame_len;
            TYPE:    tx_byte = frame_type;
            PAYLOAD: tx_byte = frame_payload[idx];
            CRC:     tx_byte = crc_val;
            default: tx_byte = 8'h00;
        endcase
    end
endmodule
