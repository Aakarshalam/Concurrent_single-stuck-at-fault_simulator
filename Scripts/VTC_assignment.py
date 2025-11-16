import argparse
import collections
import csv
import io
import re
import sys
from typing import Dict, List, Set, Tuple, Optional, NamedTuple

# Data structures

class Gate(NamedTuple):
    gtype: str           # 'AND','OR','NOT','NAND','NOR','XOR','XNOR','DFF'
    out: str
    ins: Tuple[str, ...] # ('a','b',...) or ('a',) or ('d','clk') for DFF

class Circuit:
    def __init__(self, name: str):
        self.name = name
        self.PIs: List[str] = []
        self.POs: List[str] = []
        self.wires: Set[str] = set()
        self.gates: List[Gate] = []
        self.drivers: Dict[str, Gate] = {}
        self.fanout: Dict[str, List[str]] = collections.defaultdict(list)
        self.state_nodes: List[str] = []       # Q outputs of DFFs
        self.dff_defs: List[Gate] = []         # DFFs (Q out, (D,CLK))

    def finalize(self):
        self.drivers.clear()
        self.fanout.clear()
        self.state_nodes.clear()
        self.dff_defs.clear()
        for g in self.gates:
            self.drivers[g.out] = g
            if g.gtype == 'DFF':
                self.state_nodes.append(g.out)
                self.dff_defs.append(g)
            for s in g.ins:
                self.fanout[s].append(g.out)

    def topo_order(self) -> List[Gate]:
        """Topological order for *combinational* portion (DFFs are cuts)."""
        comb = [g for g in self.gates if g.gtype != 'DFF']
        out_nets = {g.out for g in comb}
        indeg = {g: 0 for g in comb}
        for g in comb:
            for i in g.ins:
                if i in out_nets:
                    indeg[g] += 1
        q = collections.deque([g for g, c in indeg.items() if c == 0])
        order = []
        while q:
            g = q.popleft()
            order.append(g)
            for h in comb:
                if g.out in h.ins:
                    indeg[h] -= 1
                    if indeg[h] == 0:
                        q.append(h)
        return order if len(order) == len(comb) else comb


# Verilog parser 

VERILOG_PRIMS = {
    'and':  'AND',  'or':   'OR',   'not':  'NOT',  'dff':  'DFF',
    'nand': 'NAND', 'nor':  'NOR',  'xor':  'XOR',  'xnor': 'XNOR',
}

def parse_verilog_structural(path: str, top: Optional[str]=None) -> Circuit:
    with open(path, 'r', encoding='utf-8-sig') as f:
        text = f.read()

    # strip comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    mods = re.findall(
        r'\bmodule\s+([A-Za-z_]\w*)\s*\((.*?)\);\s*(.*?)\bendmodule',
        text, flags=re.DOTALL | re.IGNORECASE
    )
    if not mods:
        raise ValueError("No module found in Verilog file.")
    if top:
        chosen = None
        for name, ports, body in mods:
            if name == top:
                chosen = (name, ports, body); break
        if chosen is None:
            raise ValueError(f"Top module '{top}' not found.")
    else:
        chosen = mods[-1]

    name, port_blob, body = chosen
    ckt = Circuit(name)

    def split_names_csv(s: str) -> List[str]:
        out = []
        for n in s.split(','):
            n = n.strip()
            if not n:
                continue
            n = re.sub(r'\[[^\]]+\]\s*', '', n).strip()   # drop [m:n]
            if n and n not in ('wire', 'reg'):
                out.append(n)
        return out

    def parse_ansi_ports(header: str):
        header = re.sub(r'\s+', ' ', header.strip())
        pis, pos, any_wires = [], [], set()
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
                else:
                    pis.append(name); pos.append(name)
                any_wires.add(name)
        return pis, pos, any_wires

    # legacy body decls
    body_inputs = []
    for blob in re.findall(r'\binput\s+([^;]+);', body, flags=re.IGNORECASE):
        body_inputs += split_names_csv(blob)
    body_outputs = []
    for blob in re.findall(r'\boutput\s+([^;]+);', body, flags=re.IGNORECASE):
        body_outputs += split_names_csv(blob)
    body_wires = []
    for blob in re.findall(r'\bwire\s+([^;]+);', body, flags=re.IGNORECASE):
        body_wires += split_names_csv(blob)

    header_pis, header_pos, header_wires = parse_ansi_ports(port_blob)
    ckt.PIs = list(dict.fromkeys(body_inputs + header_pis))
    ckt.POs = list(dict.fromkeys(body_outputs + header_pos))
    ckt.wires = set(body_wires) | header_wires | set(ckt.PIs) | set(ckt.POs)

    # primitive instances (variadic gates allowed)
    insts = re.findall(r'(?i)\b(and|or|not|dff|nand|nor|xor|xnor)\b\s*(?:\w+)?\s*\(([^)]+)\)\s*;', body)
    for prim, args in insts:
        args = [a.strip() for a in args.split(',') if a.strip()]
        p = VERILOG_PRIMS[prim.lower()]
        if p in ('AND','OR','NAND','NOR','XOR','XNOR'):
            if len(args) < 3:
                raise ValueError(f"{p} requires (out, in1, in2, ...)")
            g = Gate(p, args[0], tuple(args[1:]))
        elif p == 'NOT':
            if len(args) != 2:
                raise ValueError("NOT requires (out, in)")
            g = Gate('NOT', args[0], (args[1],))
        elif p == 'DFF':
            if len(args) == 3:
                d, clk, q = args
            elif len(args) == 2:
                d, q = args; clk = 'clk'
            else:
                raise ValueError("DFF requires (D, clk, Q)")
            g = Gate('DFF', q, (d, clk))
            if clk not in ckt.PIs and clk not in ckt.wires and clk not in ckt.POs:
                ckt.PIs.append(clk)
        ckt.gates.append(g)
        ckt.wires.add(g.out)
        for s in g.ins:
            if s not in ckt.PIs and s not in ckt.POs:
                ckt.wires.add(s)

    ckt.finalize()
    return ckt


# Fault model & collapsing

class Fault(NamedTuple):
    net: str
    sa: int      # 0 or 1
    idx: int     # reindexed after collapsing

def not_chain_collapse(c: Circuit, faults: List[Fault]) -> List[Fault]:
    """
    Collapse across single-fanout NOT chains:
      ... -- A --NOT--> B --NOT--> C ...
    Map (sa@B) to opposite sa@A recursively if NOT outputs have single fanout.
    """
    not_out_to_in = {}
    for g in c.gates:
        if g.gtype == 'NOT' and len(c.fanout[g.out]) == 1:
            not_out_to_in[g.out] = g.ins[0]

    out = []
    for f in faults:
        net, sa = f.net, f.sa
        visited = set()
        while net in not_out_to_in and net not in visited:
            visited.add(net)
            net = not_out_to_in[net]
            sa ^= 1
        out.append(Fault(net, sa, f.idx))
    return out

def enumerate_faults(c: Circuit) -> List[Fault]:
    nets = set(c.wires) | set(c.PIs) | set(c.POs) | set(c.state_nodes)
    base = [Fault(n, 0, i) for i, n in enumerate(sorted(nets))]
    base += [Fault(n, 1, i+len(nets)) for i, n in enumerate(sorted(nets))]
    return base

def dominance_collapse(c: Circuit, faults: List[Fault]) -> List[Fault]:
    existing_pairs = {(f.net, f.sa) for f in faults}
    keep_pairs = set(existing_pairs)

    def drop(pair): 
        if pair in keep_pairs: keep_pairs.remove(pair)
    def keep(pair):
        keep_pairs.add(pair)

    for g in c.gates:
        if g.gtype in ('AND','NAND','OR','NOR'):
            ins = list(g.ins)
            fan1 = [i for i in ins if len(c.fanout.get(i, [])) == 1]
            any_fan1 = len(fan1) > 0

            if g.gtype == 'AND':
                keep((g.out, 1))
                if any_fan1: drop((g.out, 0))
                for i in fan1: drop((i, 1))
                # inputs sa-0 equivalent (only among fan1) -> keep the first, drop the rest
                for i in fan1[1:]: drop((i, 0))

            elif g.gtype == 'OR':
                keep((g.out, 0))
                if any_fan1: drop((g.out, 1))
                for i in fan1: drop((i, 0))
                for i in fan1[1:]: drop((i, 1))

            elif g.gtype == 'NAND':
                keep((g.out, 0))
                if any_fan1: drop((g.out, 1))
                for i in fan1: drop((i, 0))
                for i in fan1[1:]: drop((i, 1))

            elif g.gtype == 'NOR':
                keep((g.out, 1))
                if any_fan1: drop((g.out, 0))
                for i in fan1: drop((i, 1))
                for i in fan1[1:]: drop((i, 0))

    kept = sorted([p for p in keep_pairs], key=lambda x: (x[0], x[1]))
    out: List[Fault] = []
    for idx, (net, sa) in enumerate(kept):
        out.append(Fault(net, sa, idx))
    return out


# Test vector reader

def read_vectors(path: str, pins: List[str]) -> List[Dict[str, int]]:
    vectors: List[Dict[str, int]] = []

    with open(path, 'r', encoding='utf-8-sig') as f:
        peek = f.read(1024)

    if '=' not in peek:
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any((x or '').strip() for x in r)]
        if not rows:
            return []
        header = [h.strip() for h in rows[0]]
        for r in rows[1:]:
            vec = {pin: 0 for pin in pins}
            for h, v in zip(header, r):
                h = h.strip()
                if h in vec:
                    vec[h] = 1 if str(v).strip() in ('1','H','h','true','True') else 0
            vectors.append(vec)
        return vectors

    # key=value format
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = (line or '').strip()
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


def all_mask(bits: int) -> int: return (1 << bits) - 1
def bit_not(x: int, bits: int) -> int: return (~x) & all_mask(bits)
def encode_const(v: int, bits: int) -> int: return all_mask(bits) if v else 0

def gate_eval_bits(g: Gate, getv, bits: int) -> int:
    if g.gtype == 'AND':
        acc = encode_const(1, bits)
        for i in g.ins: acc &= getv(i)
        return acc
    if g.gtype == 'OR':
        acc = 0
        for i in g.ins: acc |= getv(i)
        return acc
    if g.gtype == 'NAND':
        acc = encode_const(1, bits)
        for i in g.ins: acc &= getv(i)
        return bit_not(acc, bits)
    if g.gtype == 'NOR':
        acc = 0
        for i in g.ins: acc |= getv(i)
        return bit_not(acc, bits)
    if g.gtype == 'XOR':
        acc = 0
        for i in g.ins: acc ^= getv(i)
        return acc
    if g.gtype == 'XNOR':
        acc = 0
        for i in g.ins: acc ^= getv(i)
        return bit_not(acc, bits)
    if g.gtype == 'NOT':
        return bit_not(getv(g.ins[0]), bits)
    raise RuntimeError("Non-combinational gate in comb eval")


# Event-driven concurrent simulation

def simulate_event_driven(c: Circuit,
                          faults: List[Fault],
                          vectors: List[Dict[str, int]],
                          out_path: str,
                          batch_bits: int = 63) -> None:
    """
    Event-driven, bit-parallel simulator with fault dropping.
    - Cache prev bit-vectors per batch (fault slice). If batch composition is unchanged,
      only nets impacted by source changes re-evaluate and propagate.
    """
    undetected: Dict[int, Fault] = {i: f for i, f in enumerate(faults)}
    detected_by: Dict[int, List[int]] = collections.defaultdict(list)

    # Golden state (bit0) for DFFs
    state: Dict[str, int] = {q: 0 for q in c.state_nodes}

    # Precompute comb topo order and fanin->gate map
    order = c.topo_order()
    net_to_gates: Dict[str, List[Gate]] = collections.defaultdict(list)
    for g in order:
        for i in g.ins:
            net_to_gates[i].append(g)

    batch_cache: Dict[Tuple[int, ...], Dict[str, int]] = {}
    prev_clk = 0

    for v_idx, vec in enumerate(vectors, start=1):
        clk_now = vec.get('clk', prev_clk)
        rising = (prev_clk == 0 and clk_now == 1)
        prev_clk = clk_now

        keys = list(undetected.keys())
        if not keys:
            break

        for start in range(0, len(keys), batch_bits):
            kslice = keys[start:start+batch_bits]
            bits = 1 + len(kslice)
            ALL = all_mask(bits)
            FAULT_MASK = ALL ^ 1

            # per-net fault injection masks
            inj0: Dict[str, int] = collections.defaultdict(int)
            inj1: Dict[str, int] = collections.defaultdict(int)
            for bpos, k in enumerate(kslice, start=1):
                f = undetected[k]
                if f.sa == 0: inj0[f.net] |= (1 << bpos)
                else:         inj1[f.net] |= (1 << bpos)

            key = tuple(kslice)
            prev = batch_cache.get(key, {})    # previous bit-vectors for this batch
            val: Dict[str, int] = dict(prev)

            def getv(n: str) -> int:
                return val.get(n, 0)

            # Queue of affected gates
            q = collections.deque()
            enqueued: Set[Gate] = set()

            # Seed changes: PIs + DFF Qs
            changed_nets: List[str] = []

            # PIs
            for pi in c.PIs:
                gv = vec.get(pi, 0)
                x  = encode_const(gv, bits)
                if inj1[pi]: x |= inj1[pi]
                if inj0[pi]: x &= (ALL ^ inj0[pi])
                if val.get(pi, None) != x:
                    val[pi] = x
                    changed_nets.append(pi)

            # DFF Qs as sources
            for qn in c.state_nodes:
                gv = state[qn]
                x  = encode_const(gv, bits)
                if inj1[qn]: x |= inj1[qn]
                if inj0[qn]: x &= (ALL ^ inj0[qn])
                if val.get(qn, None) != x:
                    val[qn] = x
                    changed_nets.append(qn)

            # Enqueue gates fed by changed nets
            for n in changed_nets:
                for g in net_to_gates.get(n, []):
                    if g not in enqueued:
                        enqueued.add(g)
                        q.append(g)

            # If no cache (first time for this batch), compute full comb once
            if not prev:
                enqueued = set(order)
                q = collections.deque(order)

            # Propagate changes
            while q:
                g = q.popleft()
                x = gate_eval_bits(g, getv, bits)
                if inj1[g.out]: x |= inj1[g.out]
                if inj0[g.out]: x &= (ALL ^ inj0[g.out])
                if val.get(g.out, None) != x:
                    val[g.out] = x
                    for h in net_to_gates.get(g.out, []):
                        if h not in enqueued:
                            enqueued.add(h)
                            q.append(h)

            # Rising-edge: update Qs (becomes observable immediately)
            if c.dff_defs and rising:
                changed = []
                for dff in c.dff_defs:
                    dnet, clknet = dff.ins
                    x = val.get(dnet, 0)
                    if inj1[dff.out]: x |= inj1[dff.out]
                    if inj0[dff.out]: x &= (ALL ^ inj0[dff.out])
                    if val.get(dff.out, None) != x:
                        val[dff.out] = x
                        changed.append(dff.out)
                # Propagate from changed Qs
                for n in changed:
                    for g in net_to_gates.get(n, []):
                        if g not in enqueued:
                            enqueued.add(g); q.append(g)
                while q:
                    g = q.popleft()
                    x = gate_eval_bits(g, getv, bits)
                    if inj1[g.out]: x |= inj1[g.out]
                    if inj0[g.out]: x &= (ALL ^ inj0[g.out])
                    if val.get(g.out, None) != x:
                        val[g.out] = x
                        for h in net_to_gates.get(g.out, []):
                            if h not in enqueued:
                                enqueued.add(h); q.append(h)

            # Detection at POs
            batch_detect_bits = 0
            for po in c.POs:
                x = val.get(po, 0)
                good = (x & 1)
                rep  = ALL if good else 0
                diff = (x ^ rep) & FAULT_MASK
                batch_detect_bits |= diff

            if batch_detect_bits:
                w = batch_detect_bits
                while w:
                    b = (w & -w); w ^= b
                    bpos = (b.bit_length() - 1)
                    k = kslice[bpos - 1]
                    detected_by[k].append(v_idx)
                # fault dropping
                for bpos, k in enumerate(kslice, start=1):
                    if (batch_detect_bits >> bpos) & 1:
                        undetected.pop(k, None)

            # Commit golden state on rising edge
            if c.dff_defs and rising:
                for dff in c.dff_defs:
                    qv = val.get(dff.out, encode_const(state[dff.out], bits))
                    state[dff.out] = 1 if (qv & 1) else 0

            # Snapshot cache for event-driven next vector (same batch)
            batch_cache[key] = val

 
    # Write report
    total_faults = len(faults)
    detected_count = len(set(detected_by.keys()))
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
            ff = faults[k]
            first_v = detected_by[k][0]
            f.write(f"  id={k:4d} : {ff.net} s-a-{ff.sa} @ v{first_v}\n")

        f.write("\nUndetected Faults:\n")
        for k in undetected_ids:
            ff = faults[k]
            f.write(f"  id={k:4d} : {ff.net} s-a-{ff.sa}\n")

    print(f"[OK] Report written to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Event-driven, bit-parallel stuck-at fault simulator")
    ap.add_argument('--verilog', help='Path to structural Verilog file')
    ap.add_argument('--top', help='Top module name')
    ap.add_argument('--vectors', help='Path to vectors file (CSV header or key=value lines)')
    ap.add_argument('--out', default='report.txt', help='Output report path')
    ap.add_argument('--batch_bits', type=int, default=63, help='Faults per batch (excludes bit0)')
    ap.add_argument('--no_dom_collapse', action='store_true', help='Disable dominance/equivalence collapse')
    ap.add_argument('--selftest', action='store_true', help='Run built-in self-test with example module')
    args = ap.parse_args()

    if args.selftest:
        verilog = """\
module something(A,B,Sel,y);
input A, B, Sel;
output y;
wire w1, w2, w3;
nand (w1, A, B);
or   (w2, B, Sel);
xor  (w3, w1, w2);
not  (y,  w3);
endmodule
"""
        with open('something.v','w') as f: f.write(verilog)
        with open('vectors.csv','w', newline='') as f:
            w = csv.writer(f); w.writerow(['A','B','Sel'])
            for A in (0,1):
                for B in (0,1):
                    for Sel in (0,1):
                        w.writerow([A,B,Sel])
        args.verilog='something.v'; args.top='something'; args.vectors='vectors.csv'; args.out='report.txt'

    if not args.verilog or not args.vectors:
        print("Provide --verilog and --vectors (or use --selftest).", file=sys.stderr)
        sys.exit(2)

    c = parse_verilog_structural(args.verilog, top=args.top)
    c.finalize()

    faults = enumerate_faults(c)
    faults = not_chain_collapse(c, faults)
    if not args.no_dom_collapse:
        faults = dominance_collapse(c, faults)

    pin_list = list(dict.fromkeys(c.PIs + c.POs + c.state_nodes + ['clk']))
    vecs = read_vectors(args.vectors, pin_list)
    if not vecs:
        print("No vectors found.", file=sys.stderr)
        sys.exit(2)

    simulate_event_driven(c, faults, vecs, args.out, batch_bits=args.batch_bits)

if __name__ == '__main__':
    main()
