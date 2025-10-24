# Concurrent Single Stuck-at-Fault Simulator  

This project implements a **concurrent fault simulator** to detect **single stuck-at faults** in digital circuits.  
Given a structural Verilog file and a set of test vectors, the tool evaluates fault effects and generates a **fault coverage report**

# Features  
- Detection of **single stuck-at-0** and **stuck-at-1** faults  
- Works on **structural Verilog netlists**  
- Accepts a **set of test vectors** as input  
- Generates a detailed **fault coverage report** (detected vs. undetected faults)  

# Usage
Clone the repository and run the following commands in Command Line / Terminal to generate reports
- python generate_vectors.py
- python VTC_assignment.py --verilog mux2to1.v --top mux2to1 --vectors vectors.csv --out report.txt
- python VTC_assignment.py --verilog seq_ckt.v --top seq_ckt --vectors vectors_seq.csv --out report.txt

This project is still **a work in progress**
