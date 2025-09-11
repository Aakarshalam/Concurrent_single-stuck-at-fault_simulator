#VTC Assignment
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent (bit-parallel) single stuck-at fault simulator
for small structural Verilog circuits (AND/OR/NOT/DFF subset).

- Reads a structural Verilog file (one top module).
- Prepares a collapsed list of single stuck-at faults (basic FFR & NOT-chain collapse).
- Reads test vectors from a simple text/CSV file.
- Simulates good and faulty circuits concurrently (bit-parallel) per vector.
- Emits coverage statistics and per-fault detection summary.

Usage:
  python faultsim.py --verilog mux2to1.v --top mux2to1 --vectors vectors.txt --out report.txt
  python faultsim.py --verilog seq_ckt.v --top seq_ckt --vectors vectors_seq.txt --out report.txt
"""

import argparse
import collections
import csv
import os
import re
import sys
from typing import Dict, List, Set, Tuple, Optional, NamedTuple

# ---------------------------
# Data structures
# ---------------------------

class Gate(NamedTuple):
    gtype: str           # 'AND','OR','NOT','DFF'
    out: str
    ins: Tuple[str, ...] # ('a','b') or ('a',) or ('d','clk') for DFF (clk optional in eval, we handle state)

class Circuit:
    def __init__(self, name: str):
        self.name = name
        self.PIs: List[str] = []
        self.POs: List[str] = []
        self.wires: Set[str] = set()
        self.gates: List[Gate] = []
        self.drivers: Dict[str, Gate] = {}     # net -> driving gate
        self.fanout: Dict[str, List[str]] = collections.defaultdict(list)  # net -> list of sinks (by net)
        self.state_nodes: List[str] = []       # Q outputs of DFFs (stateful outputs)
        self.dff_defs: List[Gate] = []         # DFF gates, ins=(D, CLK), out=Q

    def finalize(self):
        # record drivers and fanout
        for g in self.gates:
            if g.gtype != 'DFF':
                self.drivers[g.out] = g
            else:
                # Treat DFF as a stateful element: the Q "driver" is the flop
                self.drivers[g.out] = g
                self.state_nodes.append(g.out)
                self.dff_defs.append(g)
            for s in g.ins:
                self.fanout[s].append(g.out)

    def topo_order(self) -> List[Gate]:
        """
        Topological order for the combinational portion.
        Treat DFF Q outputs as sources; do not "evaluate" DFF as a combinational gate.
        """
        indeg = collections.Counter()
        nodes = set()
        comb_gates = []
        for g in self.gates:
            if g.gtype == 'DFF':
                continue
            comb_gates.append(g)
            nodes.add(g.out)
            for i in g.ins:
                indeg[g.out] += 0  # ensure key
                indeg[g.out] += 1

        # Inputs for comb logic are PIs + DFF Qs
        sources = set(self.PIs) | set(self.state_nodes)

        # Kahn's algorithm over gate graph by output dependency
        # Build net->dependent gates map
        net_to_gates: Dict[str, List[Gate]] = collections.defaultdict(list)
        for g in comb_gates:
            for i in g.ins:
                net_to_gates[i].append(g)

        # indegree per gate = number of inputs that are driven by comb outputs (not sources)
        gate_in_cnt: Dict[Gate, int] = {}
        for g in comb_gates:
            cnt = 0
            for i in g.ins:
                if i in sources:
                    continue
                # If the input is from another gate output, count it
                if i in (gg.out for gg in comb_gates) or i in self.state_nodes:
                    # if a comb gate drives it, we count
                    if i in (gg.out for gg in comb_gates):
                        cnt += 1
                # else assume source/PI
            gate_in_cnt[g] = cnt

        q = collections.deque([g for g, c in gate_in_cnt.items() if c == 0])
        order = []
        while q:
            g = q.popleft()
            order.append(g)
            # Any gate whose inputs include g.out reduces indegree
            for h in comb_gates:
                if h in gate_in_cnt and g.out in h.ins:
                    gate_in_cnt[h] -= 1
                    if gate_in_cnt[h] == 0:
                        q.append(h)

        # Fall back if cycle (shouldn't happen for pure comb + DFF cuts)
        if len(order) != len(comb_gates):
            # naive: return in original order
            return comb_gates
        return order

# ---------------------------
# Verilog parser (tiny subset)
# ---------------------------

VERILOG_PRIMS = {
    'and': 'AND',
    'or':  'OR',
    'not': 'NOT',
    'dff': 'DFF',  # assume form: dff inst (D, clk, Q);
}

def parse_verilog_structural(path: str, top: Optional[str]=None) -> Circuit:
    with open(path, 'r', encoding='utf-8-sig') as f:
        text = f.read()

    # strip comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # split modules
    mods = re.findall(
        r'\bmodule\s+([A-Za-z_]\w*)\s*\((.*?)\);\s*(.*?)\bendmodule',
        text, flags=re.DOTALL | re.IGNORECASE
    )
    if not mods:
        raise ValueError("No module found in Verilog file.")

    chosen = None
    if top:
        for name, ports, body in mods:
            if name == top:
                chosen = (name, ports, body); break
        if chosen is None:
            raise ValueError(f"Top module '{top}' not found in file.")
    else:
        chosen = mods[-1]

    name, port_blob, body = chosen
    ckt = Circuit(name)

    # ---------- helpers ----------
    def split_names_csv(s: str) -> List[str]:
        out = []
        for n in s.split(','):
            n = n.strip()
            if not n:
                continue
            # drop ranges like [3:0]
            n = re.sub(r'\[[^\]]+\]\s*', '', n).strip()
            if n and n not in ('wire', 'reg'):
                out.append(n)
        return out

    def parse_ansi_ports(header: str):
        header = re.sub(r'\s+', ' ', header.strip())
        pis, pos, any_wires = [], [], set()
        # Capture chunks between direction keywords
        for dirw, names in re.findall(
            r'\b(input|output|inout)\b\s+([^()]+?)(?=(?:\binput\b|\boutput\b|\binout\b|$))',
            header, flags=re.IGNORECASE
        ):
            names = names.strip().rstrip(',')
            for name in split_names_csv(names):
                if dirw.lower() == 'input':
                    pis.append(name)
                elif dirw.lower() == 'output':
                    pos.append(name)
                else:  # inout
                    pis.append(name); pos.append(name)
                any_wires.add(name)
        return pis, pos, any_wires

    # ---------- parse ANSI header + legacy body decls ----------
    # Body decls like: input A,B; output Y; wire w1,w2;
    body_inputs = []
    for blob in re.findall(r'\binput\s+([^;]+);', body, flags=re.IGNORECASE):
        body_inputs += split_names_csv(blob)
    body_outputs = []
    for blob in re.findall(r'\boutput\s+([^;]+);', body, flags=re.IGNORECASE):
        body_outputs += split_names_csv(blob)
    body_wires = []
    for blob in re.findall(r'\bwire\s+([^;]+);', body, flags=re.IGNORECASE):
        body_wires += split_names_csv(blob)

    # ANSI header (input/output in the port list)
    header_pis, header_pos, header_wires = parse_ansi_ports(port_blob)

    # Final PIs/POs/wires
    ckt.PIs = list(dict.fromkeys(body_inputs + header_pis))
    ckt.POs = list(dict.fromkeys(body_outputs + header_pos))
    ckt.wires = set(body_wires) | header_wires | set(ckt.PIs) | set(ckt.POs)

    # ---------- primitive instances ----------
    insts = re.findall(r'(?i)\b(and|or|not|dff)\b\s*(?:\w+)?\s*\(([^)]+)\)\s*;', body)
    for prim, args in insts:
        args = [a.strip() for a in args.split(',') if a.strip()]
        p = VERILOG_PRIMS[prim.lower()]
        if p in ('AND','OR'):
            if len(args) < 3:
                raise ValueError(f"{p} requires (out, in1, in2)")
            g = Gate(p, args[0], tuple(args[1:]))
        elif p == 'NOT':
            if len(args) != 2:
                raise ValueError("NOT requires (out, in)")
            g = Gate(p, args[0], (args[1],))
        elif p == 'DFF':
            if len(args) == 3:
                d, clk, q = args
            elif len(args) == 2:
                d, q = args; clk = 'clk'
            else:
                raise ValueError("DFF requires (D, clk, Q)")
            g = Gate(p, q, (d, clk))
            if clk not in ckt.PIs and clk not in ckt.wires and clk not in ckt.POs:
                ckt.PIs.append(clk)
        else:
            raise ValueError(f"Unsupported primitive: {prim}")
        ckt.gates.append(g)
        ckt.wires.add(g.out)
        for s in g.ins:
            if s not in ckt.PIs and s not in ckt.POs:
                ckt.wires.add(s)

    ckt.finalize()
    return ckt
# ---------------------------
# Fault model & collapsing
# ---------------------------

class Fault(NamedTuple):
    net: str
    sa: int      # 0 or 1
    idx: int     # stable index in global list

def not_chain_collapse(c: Circuit, faults: List[Fault]) -> List[Fault]:
    """
    Collapse stuck-at faults across single-fanout NOT chains:
    A --NOT--> B --NOT--> C ...
    sa@B maps to opposite sa@A if NOT has fanout==1, recursively.
    """ 
    out_faults = []
    # quick lookup: for NOT outputs with fanout==1, map to input
    not_out_to_in = {}
    for g in c.gates:
        if g.gtype == 'NOT':
            if len(c.fanout[g.out]) == 1:  # single fanout => safe to map
                not_out_to_in[g.out] = g.ins[0]

    for f in faults:
        net = f.net
        sa = f.sa
        # Follow NOT chain backwards while legal
        visited = set()
        while net in not_out_to_in and net not in visited:
            visited.add(net)
            net = not_out_to_in[net]
            sa ^= 1  # invert polarity
        out_faults.append(Fault(net, sa, f.idx))
    # Deduplicate by (net, sa, idx keeps identity but we remove logical dups down below when packing)
    # We keep per-idx identity because detection reporting is by idx.
    return out_faults

def enumerate_faults(c: Circuit, collapse=True) -> List[Fault]:
    """
    Create single stuck-at faults on all nets (including PIs and state nodes).
    Basic collapse: NOT-chain only (safe & simple).
    """
    nets = set(c.wires) | set(c.PIs) | set(c.POs) | set(c.state_nodes)
    base = [Fault(n, 0, i) for i, n in enumerate(sorted(nets))]
    base += [Fault(n, 1, i+len(nets)) for i, n in enumerate(sorted(nets))]
    if collapse:
        return not_chain_collapse(c, base)
    return base

# ---------------------------
# Test vector reader
# ---------------------------

def read_vectors(path: str, pins: List[str]) -> List[Dict[str, int]]:
    """
    Supported formats:
    - Space or comma separated key=value pairs, e.g.:
        A=0 B=1 Sel=1
      or
        A, B, Sel
        0,1,1
        1,0,0
    - CSV with header of pin names.
    Missing pins default to 0. Extra fields are ignored.

    For sequential circuits, include "clk" transitions if you want rising-edge sampling.
    """
    vectors: List[Dict[str, int]] = []

    # Try CSV with header first
    with open(path, 'r') as f:
        peek = f.read(1024)
    if '=' not in peek:
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(x.strip() for x in r)]
        if not rows:
            return []
        header = [h.strip() for h in rows[0]]
        body = rows[1:]
        for r in body:
            vec = {pin: 0 for pin in pins}
            for h, v in zip(header, r):
                h = h.strip()
                if h in vec:
                    vec[h] = 1 if str(v).strip() in ('1','H','h','true','True') else 0
            vectors.append(vec)
        return vectors

    # Fallback: kv pairs per line
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = re.split(r'[,\s]+', line)
            vec = {pin: 0 for pin in pins}
            for p in parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    k = k.strip(); v = v.strip()
                    if k in vec:
                        vec[k] = 1 if v in ('1','H','h','true','True') else 0
            vectors.append(vec)
    return vectors

# ---------------------------
# Bit-parallel simulation helpers
# ---------------------------

def all_mask(bits: int) -> int:
    return (1 << bits) - 1

def bit_not(x: int, bits: int) -> int:
    return (~x) & all_mask(bits)

def encode_const(v: int, bits: int) -> int:
    return all_mask(bits) if v else 0

def gate_eval(g: Gate, getv, bits: int) -> int:
    if g.gtype == 'AND':
        acc = encode_const(1, bits)
        for i in g.ins:
            acc &= getv(i)
        return acc
    if g.gtype == 'OR':
        acc = 0
        for i in g.ins:
            acc |= getv(i)
        return acc
    if g.gtype == 'NOT':
        return bit_not(getv(g.ins[0]), bits)
    raise RuntimeError("Combinational eval called on non-comb gate")

# ---------------------------
# Fault simulation (bit-parallel)
# ---------------------------

def simulate(c: Circuit,
             faults: List[Fault],
             vectors: List[Dict[str, int]],
             out_path: str,
             batch_bits: int = 63) -> None:
    """
    Runs simulation and writes report.
    batch_bits: number of fault bits per word (excluding bit0 which is good circuit).
                63 fits into 64-bit-ish mental model; Python ints allow larger, but 63 is fine.
    """
    # Prepare batches of faults
    undetected: Dict[int, Fault] = {i: f for i, f in enumerate(faults)}
    detected_by: Dict[int, List[int]] = collections.defaultdict(list)  # fault_idx -> list of vector indices (1-based)

    # Initial DFF state (all zeros unless user provides Q in vectors; Q as PO will reflect state)
    state: Dict[str, int] = {q: 0 for q in c.state_nodes}

    # Precompute topo order
    order = c.topo_order()

    # Evaluate each test vector
    for v_idx, vec in enumerate(vectors, start=1):
        # Rising-edge detection for sequential update
        prev_clk = getattr(simulate, "_prev_clk", 0)
        clk_now = vec.get('clk', prev_clk)
        rising_edge = (prev_clk == 0 and clk_now == 1)
        simulate._prev_clk = clk_now  # store for next call

        # Batches
        keys = list(undetected.keys())
        if not keys:
            break
        for start in range(0, len(keys), batch_bits):
            kslice = keys[start:start+batch_bits]
            # bit width = 1 (golden) + faults in this batch
            bits = 1 + len(kslice)
            ALL = all_mask(bits)
            FAULT_MASK = ALL ^ 1  # all bits except bit0

            # Prepare injection masks per net
            inj0: Dict[str, int] = collections.defaultdict(int)  # force 0 bits
            inj1: Dict[str, int] = collections.defaultdict(int)  # force 1 bits

            # Map per-bit position
            fault_to_bit: Dict[int, int] = {}
            for bpos, k in enumerate(kslice, start=1):
                f = undetected[k]
                fault_to_bit[k] = bpos
                if f.sa == 0:
                    inj0[f.net] |= (1 << bpos)
                else:
                    inj1[f.net] |= (1 << bpos)

            # Signal values for this batch (bit-parallel)
            val: Dict[str, int] = {}

            # Load PIs
            for pi in c.PIs:
                gv = vec.get(pi, 0)
                x = encode_const(gv, bits)
                # inject if fault site at this PI
                if inj1[pi]:
                    x |= inj1[pi]
                if inj0[pi]:
                    x &= (ALL ^ inj0[pi])
                val[pi] = x

            # Load DFF Qs as sources (state)
            for q in c.state_nodes:
                gv = state[q]
                x = encode_const(gv, bits)
                if inj1[q]:
                    x |= inj1[q]
                if inj0[q]:
                    x &= (ALL ^ inj0[q])
                val[q] = x

            # Evaluate combinational in topo order
            def getv(n: str) -> int:
                if n not in val:
                    # Undriven nets default to 0
                    val[n] = 0
                return val[n]

            for g in order:
                x = gate_eval(g, getv, bits)
                # inject at this net if needed
                if inj1[g.out]:
                    x |= inj1[g.out]
                if inj0[g.out]:
                    x &= (ALL ^ inj0[g.out])
                val[g.out] = x

            # Compute DFF next-state on rising edge (good + faults)
            if c.dff_defs:
                for dff in c.dff_defs:
                    dnet, clknet = dff.ins
                    if rising_edge:
                        # New Q becomes D, but also respect faults at Q (already applied above
                        # to current Q for combinational; now apply to next state as well)
                        x = val.get(dnet, 0)
                        # If there are faults *at Q* in this batch, re-apply when we commit state
                        if inj1[dff.out]:
                            x |= inj1[dff.out]
                        if inj0[dff.out]:
                            x &= (ALL ^ inj0[dff.out])
                        val[dff.out] = x  # observable after edge
                    # else keep current Q in val[dff.out] (already loaded)

            # Determine detection at POs
            batch_detect_bits = 0
            for po in c.POs:
                x = val.get(po, 0)
                good = (x & 1)
                rep = ALL if good else 0
                diff = (x ^ rep) & FAULT_MASK
                batch_detect_bits |= diff

            # Record detections and update state (observable Q after edge)
            if batch_detect_bits:
                # For each bit set, mark detected
                w = batch_detect_bits
                while w:
                    b = (w & -w)
                    w ^= b
                    bpos = (b.bit_length() - 1)
                    k = kslice[bpos-1]  # map back to fault id
                    detected_by[k].append(v_idx)
                # remove detected from undetected
                for bpos, k in enumerate(kslice, start=1):
                    if (batch_detect_bits >> bpos) & 1:
                        undetected.pop(k, None)

            # Commit sequential state (use good bit0 of Q after edge)
            if c.dff_defs and rising_edge:
                for dff in c.dff_defs:
                    qv = val.get(dff.out, encode_const(state[dff.out], bits))
                    state[dff.out] = 1 if (qv & 1) else 0

        # end batches
    # end vectors

    # ---------------------------
    # Write report
    # ---------------------------
    total_faults = len(faults)
    detected_count = len(detected_by)
    undetected_ids = sorted(set(range(total_faults)) - set(detected_by.keys()))

    with open(out_path, 'w') as f:
        f.write(f"Fault Simulation Report\n")
        f.write(f"=======================\n")
        f.write(f"Top module       : {c.name}\n")
        f.write(f"Primary Inputs   : {', '.join(c.PIs)}\n")
        f.write(f"Primary Outputs  : {', '.join(c.POs)}\n")
        if c.state_nodes:
            f.write(f"State Nodes (Q)  : {', '.join(c.state_nodes)}\n")
        f.write(f"Vectors simulated: {len(vectors)}\n")
        f.write(f"Faults (total)   : {total_faults}\n")
        f.write(f"Detected         : {detected_count}\n")
        cov = 100.0 * detected_count / total_faults if total_faults else 0.0
        f.write(f"Coverage         : {cov:.2f}%\n\n")

        f.write("Detected Faults (net, sa, first_detected_at_vector):\n")
        for k in sorted(detected_by.keys()):
            # Find the fault as originally enumerated (pre-collapse identity kept by idx)
            ff = faults[k]
            first_v = detected_by[k][0]
            f.write(f"  id={k:4d} : {ff.net} s-a-{ff.sa} @ v{first_v}\n")
        f.write("\nUndetected Faults:\n")
        for k in undetected_ids:
            ff = faults[k]
            f.write(f"  id={k:4d} : {ff.net} s-a-{ff.sa}\n")

    print(f"[OK] Report written to: {out_path}")

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Concurrent (bit-parallel) stuck-at fault simulator")
    ap.add_argument('--verilog', required=True, help='Path to structural Verilog file')
    ap.add_argument('--top', required=False, help='Top module name (defaults to last module in file)')
    ap.add_argument('--vectors', required=True, help='Path to vectors file (CSV header or key=value lines)')
    ap.add_argument('--out', default='report.txt', help='Output report path')
    ap.add_argument('--no_collapse', action='store_true', help='Disable simple collapse')
    args = ap.parse_args()

    c = parse_verilog_structural(args.verilog, top=args.top)
    c.finalize()

    faults = enumerate_faults(c, collapse=not args.no_collapse)

    # Build pin list for vector reader
    pin_list = list(dict.fromkeys(c.PIs + c.POs + c.state_nodes + ['clk']))  # stable order + clk if used
    vecs = read_vectors(args.vectors, pin_list)

    if not vecs:
        print("No vectors found. Provide a CSV (with header) or lines of key=value.", file=sys.stderr)
        sys.exit(2)

    simulate(c, faults, vecs, args.out)

if __name__ == '__main__':
    main()
