package fxp_pkg;

  // Saturate a signed value 'x' into N-bit signed range.
  function automatic logic signed [N-1:0] sat_signed #(int N=8) (input logic signed [31:0] x);
    logic signed [31:0] maxv;
    logic signed [31:0] minv;
    begin
      maxv = (1 <<< (N-1)) - 1;
      minv = -(1 <<< (N-1));
      if (x > maxv) sat_signed = logic'(maxv[N-1:0]);
      else if (x < minv) sat_signed = logic'(minv[N-1:0]);
      else sat_signed = logic'(x[N-1:0]);
    end
  endfunction

  // Round-to-nearest for right shift by S bits (S>=1).
  // Adds 0.5 LSB before shift: x + (1<<(S-1))
  function automatic logic signed [31:0] rshift_round (input logic signed [31:0] x, input int S);
    logic signed [31:0] add;
    begin
      if (S <= 0) rshift_round = x;
      else begin
        add = (1 <<< (S-1));
        rshift_round = (x + add) >>> S;
      end
    end
  endfunction

endpackage
