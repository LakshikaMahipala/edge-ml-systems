// Fixed-point helper functions shared by all INT8 kernels.
//
// Requantization used everywhere:
//     y_int8 = saturate_int8( (acc + 2^(SHIFT-1)) >>> SHIFT )
// i.e. multiply the int32 accumulator by 2^-SHIFT with round-half-up, then clip.
// The bit-exact Python twin of every function lives in golden/int8_ref.py.
package fxp_pkg;

    // Arithmetic right shift by s bits with rounding (round half up).
    function automatic logic signed [31:0] rshift_round(input logic signed [31:0] x,
                                                         input int s);
        if (s <= 0) return x;
        return (x + (32'sd1 <<< (s - 1))) >>> s;
    endfunction

    // Clip a signed 32-bit value to the range of an n-bit signed integer.
    // The result is still 32 bits wide; take the low n bits to store it.
    function automatic logic signed [31:0] saturate(input logic signed [31:0] x, input int n);
        logic signed [31:0] max_v, min_v;
        max_v = (32'sd1 <<< (n - 1)) - 32'sd1;
        min_v = -(32'sd1 <<< (n - 1));
        if (x > max_v) return max_v;
        if (x < min_v) return min_v;
        return x;
    endfunction

    // Full INT8 requantization: round-shift, then clip to [-128, 127].
    function automatic logic signed [7:0] requant_int8(input logic signed [31:0] acc,
                                                       input int shift);
        logic signed [31:0] clipped;
        clipped = saturate(rshift_round(acc, shift), 8);
        return clipped[7:0];
    endfunction

endpackage
