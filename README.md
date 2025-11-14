# Concurrent Single Stuck-at-Fault Simulator  

This project implements a **concurrent fault simulator** to detect **single stuck-at faults** in digital circuits.  
Given a structural Verilog file and a set of test vectors, the tool evaluates fault effects and generates a **fault coverage report**

# Features  
- Detection of **single stuck-at-0** and **stuck-at-1** faults  
- Works on **structural Verilog netlists**  
- Accepts a **set of test vectors** as input  
- Generates a detailed **fault coverage report** (detected vs. undetected faults)  

# Usage
Clone the repository and run the following commands in the respective directory in Command Line / Terminal to generate reports
1) Combinational circuits
```bash
- python generate_vectors.py --verilog something.v --top something --out vectors.csv
- python VTC_assignment.py --verilog something.v --top something --vectors vectors.csv --out something_report.txt
```
2) Sequential circuits
```bash
- python generate_vectors.py --verilog seq_ckt.v --top seq_ckt --out vectors.csv
- python VTC_assignment.py --verilog seq_ckt.v --top seq_ckt --vectors vectors.csv --out seq_report.txt
```

