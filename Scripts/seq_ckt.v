module seq_ckt(clk, rst, d, q, qbar);
    input  clk, rst, d;
    output q, qbar;

    wire n_rst;
    wire d_masked;

    not  (n_rst, rst);   
    and  (d_masked, d, n_rst); 

    dff  (d_masked, clk, q);    

    not  (qbar, q);             
endmodule
