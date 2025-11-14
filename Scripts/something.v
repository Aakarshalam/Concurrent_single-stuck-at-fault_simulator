module something(A,B,Sel,y);

input A, B, Sel;
output y;

wire w1, w2, w3;
nand (w1, A, B);
or (w2, B, Sel);
xor (w3, w1, w2);
not (y, w3);

endmodule