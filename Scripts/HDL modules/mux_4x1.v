module mux_4x1(a,b,c,d,s1,s2,y);

input a,b,c,d,s1,s2;
output y;

wire w1, w3, w4, w5, w6, w7, w8, w9,w10,w11;

and(w3, a, s1);
not(w1, s1);
and(w4, w1, b);
or ( w7, w3, w4);
and(w5, c, s1);
and(w6, w1, d);
or ( w8, w5, w6);
and(w10, s2, w7);
not(w9, s2);
and(w11, w8, w9);
or(y, w10, w11);

endmodule