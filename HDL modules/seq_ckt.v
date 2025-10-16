module seq_ckt(A, B, clk, Q);
  input  A, B, clk;
  output Q;

  wire w1;
  and (w1, A, B);
  dff ff1 (w1, clk, Q);
endmodule

