
"""
Usage:
  python generate_vectors.py --verilog something.v --top something --out vectors.csv
  python generate_vectors.py --verilog my.v --out vectors.csv --clk-name CLK

"""

import argparse
import csv
import itertools
import re
from typing import List, Tuple, Set, Optional

def _strip_comments(text: str) -> str:
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text

def _split_names_csv(s: str) -> List[str]:
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

def _parse_ansi_ports(header: str) -> Tuple[List[str], List[str], Set[str]]:
    """Return (inputs, outputs, all_declared_names) from an ANSI-style port list."""
    header = re.sub(r'\s+', ' ', header.strip())
    pis, pos, any_wires = [], [], set()
    for dirw, names in re.findall(
        r'\b(input|output|inout)\b\s+([^()]+?)(?=(?:\binput\b|\boutput\b|\binout\b|$))',
        header, flags=re.IGNORECASE
    ):
        names = names.strip().rstrip(',')
        for name in _split_names_csv(names):
            if dirw.lower() == 'input':
                pis.append(name)
            elif dirw.lower() == 'output':
                pos.append(name)
            else:  # inout -> treat as both (conservative)
                pis.append(name); pos.append(name)
            any_wires.add(name)
    return pis, pos, any_wires

def get_module_and_ports(verilog_path: str, top: Optional[str]) -> Tuple[str, List[str], List[str]]:
    """
    Returns (module_name, inputs_in_order, outputs_in_order).
    Input order preference: body inputs first, then ANSI inputs (without duplicates),
    which tends to match how simple labs write ports.
    """
    with open(verilog_path, 'r', encoding='utf-8-sig') as f:
        text = _strip_comments(f.read())

    mods = re.findall(
        r'\bmodule\s+([A-Za-z_]\w*)\s*\((.*?)\);\s*(.*?)\bendmodule',
        text, flags=re.DOTALL | re.IGNORECASE
    )
    if not mods:
        raise ValueError("No module found in the Verilog file.")

    chosen = None
    if top:
        for name, ports, body in mods:
            if name == top:
                chosen = (name, ports, body); break
        if chosen is None:
            raise ValueError(f"Top module '{top}' not found.")
    else:
        # default to the last module
        chosen = mods[-1]

    name, port_blob, body = chosen

    # Legacy body decls
    body_inputs, body_outputs = [], []
    for blob in re.findall(r'\binput\s+([^;]+);', body, flags=re.IGNORECASE):
        body_inputs += _split_names_csv(blob)
    for blob in re.findall(r'\boutput\s+([^;]+);', body, flags=re.IGNORECASE):
        body_outputs += _split_names_csv(blob)

    # ANSI
    ansi_inputs, ansi_outputs, _ = _parse_ansi_ports(port_blob)

    # Merge, preserving order & removing dups
    def uniq(seq: List[str]) -> List[str]:
        return list(dict.fromkeys(seq))

    inputs  = uniq(body_inputs + ansi_inputs)
    outputs = uniq(body_outputs + ansi_outputs)
    return name, inputs, outputs

# Vector generation

def write_vectors_csv(out_path: str,
                      inputs: List[str],
                      clk_name: str = 'clk',
                      max_combos: int = 1 << 20,
                      force: bool = False) -> int:
    """
    Writes vectors to CSV and returns #rows.
    If clk_name present in inputs, emit (clk=0, clk=1) for each non-clk combo.
    """
    if not inputs:
        raise ValueError("No inputs detected for vector generation.")

    has_clk = any(p for p in inputs if p.lower() == clk_name.lower())
    # normalize case if someone used CLK vs clk
    if has_clk:
        # place the clock at the *end* of header for readability
        nonclk = [p for p in inputs if p.lower() != clk_name.lower()]
        header = nonclk + [clk_name]
    else:
        nonclk = inputs[:]
        header = inputs[:]

    n = len(nonclk)
    combos = 1 << n
    rows_needed = combos * (2 if has_clk else 1)
    if rows_needed > max_combos and not force:
        raise ValueError(
            f"Refusing to write {rows_needed} rows (>{max_combos}). "
            f"Use --force or raise --max-combos."
        )

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)

        for bits in itertools.product([0, 1], repeat=n):
            base = {name: val for name, val in zip(nonclk, bits)}
            if has_clk:
                # Rising-edge pair for each base combo
                row0 = [base[p] for p in nonclk] + [0]
                row1 = [base[p] for p in nonclk] + [1]
                w.writerow(row0)
                w.writerow(row1)
            else:
                row = [base[p] for p in header]
                w.writerow(row)

    return rows_needed


def main():
    ap = argparse.ArgumentParser(description="Generate exhaustive input vectors from a Verilog top module.")
    ap.add_argument('--verilog', required=True, help='Path to the Verilog file')
    ap.add_argument('--top', help='Top module name (defaults to last module in file)')
    ap.add_argument('--out', default='vectors.csv', help='Output CSV path')
    ap.add_argument('--clk-name', default='clk', help="Clock pin name (default: 'clk')")
    ap.add_argument('--max-combos', type=int, default=(1 << 20),
                    help='Safety cap on rows to emit (after clock expansion).')
    ap.add_argument('--force', action='store_true', help='Bypass safety cap.')
    args = ap.parse_args()

    mod, inputs, outputs = get_module_and_ports(args.verilog, top=args.top)
    print(f"[info] Top module : {mod}")
    print(f"[info] Inputs     : {inputs}")
    print(f"[info] Outputs    : {outputs}")
    print(f"[info] Clock name : {args.clk-name if hasattr(args, 'clk-name') else args.clk_name}")

    rows = write_vectors_csv(args.out, inputs, clk_name=args.clk_name,
                             max_combos=args.max_combos, force=args.force)
    print(f"[OK] Wrote {rows} rows to {args.out}")

if __name__ == '__main__':
    main()
