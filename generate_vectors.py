import csv

# vectors_mux.csv — all 8 input combos for A,B,Sel
with open('vectors_mux.csv', 'w', newline='') as f:
    w = csv.writer(f)             # comma-delimited, no BOM
    w.writerow(['A','B','Sel'])
    for Sel in [0,1]:
        for A in [0,1]:
            for B in [0,1]:
                w.writerow([A,B,Sel])

# vectors_seq.csv — pairs to create 0→1 rising edges on clk
seq_patterns = [(0,0),(1,0),(1,1),(0,1),(1,1)]
with open('vectors_seq.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['A','B','clk'])
    for A,B in seq_patterns:
        w.writerow([A,B,0])  # clk low
        w.writerow([A,B,1])  # rising edge -> latch D=A&B

print("Wrote vectors_mux.csv and vectors_seq.csv")
