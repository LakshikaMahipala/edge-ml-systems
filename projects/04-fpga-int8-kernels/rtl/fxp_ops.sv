// Fixed-point primitives in Q(N-1-F).F format: real value = integer / 2^F.

// Saturating add: y = clip(a + b)
module fxp_add #(
    parameter int N = 8
) (
    input  logic signed [N-1:0] a,
    input  logic signed [N-1:0] b,
    output logic signed [N-1:0] y
);
    import fxp_pkg::*;
    logic signed [31:0] a32, b32, clipped;
    always_comb begin
        a32     = a;  // signed -> wider signed: sign-extends
        b32     = b;
        clipped = saturate(a32 + b32, N);
        y       = clipped[N-1:0];
    end
endmodule

// Fixed-point multiply: the product has 2F fractional bits, so shift right by
// F (with rounding) to return to F fractional bits, then clip.
// Example (Q1.7): 0.5 * 0.5 -> 64 * 64 = 4096 -> (4096 + 64) >>> 7 = 32 = 0.25
module fxp_mul #(
    parameter int N = 8,
    parameter int F = 7
) (
    input  logic signed [N-1:0] a,
    input  logic signed [N-1:0] b,
    output logic signed [N-1:0] y
);
    import fxp_pkg::*;
    logic signed [31:0] a32, b32, clipped;
    always_comb begin
        a32     = a;
        b32     = b;
        clipped = saturate(rshift_round(a32 * b32, F), N);
        y       = clipped[N-1:0];
    end
endmodule
