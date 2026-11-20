#!/usr/bin/env python3
"""Compare the final state written by the solvers with the stored reference results.

For every final-state file (<algorithm>_<model>_u.vtk) it reports, for x, y, z and the magnitude:
  - the maximum displacement in the result and in the reference, and the difference between the two;
  - the largest node-by-node difference, which also catches fields that have the same maximum but differ elsewhere.
Values are in the units of the model file (metres for NH). "Maximum" is the largest absolute value over all nodes.

  python3 compare.py                                  every *_u.vtk in results/ against its reference
  python3 compare.py -a DJTLED                         only this solver's results
  python3 compare.py -m NH --tolerance 0.05            only this model, stricter pass mark (in %)
  python3 compare.py --pair results/DJTLED_NH_u.vtk "reference results/U_DJTLED.vtk"
  python3 compare.py --cpu-gpu                         GPU results against CPU results instead of against the reference

The reference for a result is the file in "reference results/" named U_<formulation>.vtk whose formulation the result's
name starts with, ignoring underscores: DJTLED_* is compared with U_DJTLED.vtk, ANPDJTLED_* with U_ANP_DJTLED.vtk.
A result counts as a match when its largest nodal difference is within --tolerance percent (default 0.1) of the
reference's maximum displacement magnitude. The table is also saved as comparison.csv next to the results.

With --cpu-gpu each GPU result is compared with the CPU result of the same formulation and model, e.g.
ANPDJTLED_CUDA_NH_u.vtk with ANPDJTLED_DET_OMP_NH_u.vtk. A solver is a GPU one if its source in src/ is a .cu file and a
CPU one if it is a .cpp file; two solvers have the same formulation when their names start with the same word (the part
before the first underscore). That table is saved as comparison_cpu_gpu.csv. Both results must exist first:
run.py --final on a machine with an NVIDIA GPU (or the Colab notebook) writes them.
Uses only the Python standard library.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # this script sits in the repository root
SRC = ROOT / "src"
AXES = ("x", "y", "z", "magnitude")


def read_displacements(path):
    """The displacement vectors (the POINT_DATA / VECTORS block) of a legacy ASCII VTK file: [(ux, uy, uz), ...]."""
    tokens = path.read_text().split()  # split() also copes with Windows line endings
    try:
        n = int(tokens[tokens.index("POINT_DATA") + 1])
        start = tokens.index("VECTORS") + 3  # VECTORS <name> <type>, then 3 numbers per node
        values = [float(t) for t in tokens[start:start + 3 * n]]
    except (ValueError, IndexError):
        raise ValueError(f"{path.name}: no displacement vectors found (not a legacy ASCII VTK file with POINT_DATA / VECTORS?)")
    if len(values) != 3 * n:
        raise ValueError(f"{path.name}: expected {n} displacement vectors, found {len(values) // 3}")
    return [tuple(values[i:i + 3]) for i in range(0, 3 * n, 3)]


def maxima(U):
    """Largest absolute displacement in x, y and z, and the largest magnitude."""
    return [max(abs(u[k]) for u in U) for k in range(3)] + [max(math.hypot(*u) for u in U)]


def compare(result, reference):
    """All the numbers for one result/reference pair."""
    R, F = read_displacements(result), read_displacements(reference)
    if len(R) != len(F):
        raise ValueError(f"different meshes: {result.name} has {len(R)} nodes, {reference.name} has {len(F)}")
    max_result, max_reference = maxima(R), maxima(F)
    delta = [tuple(r[k] - f[k] for k in range(3)) for r, f in zip(R, F)]
    nodal = maxima(delta)  # largest node-by-node difference per component and in magnitude
    worst_node = max(range(len(delta)), key=lambda i: math.hypot(*delta[i]))
    return {"result": result, "reference": reference, "nodes": len(R), "max_result": max_result, "max_reference": max_reference,
            "difference": [a - b for a, b in zip(max_result, max_reference)], "nodal": nodal, "worst_node": worst_node,
            "relative": nodal[3] / max_reference[3] if max_reference[3] > 0 else math.inf}


def squash(name):
    return name.replace("_", "").replace(" ", "").upper()


def find_reference(result, reference_dir):
    """The reference file whose formulation name (U_<formulation>.vtk) is the longest one the result's name starts with."""
    best = None
    for ref in sorted(reference_dir.glob("*.vtk")):
        key = squash(ref.stem[2:] if ref.stem.upper().startswith("U_") else ref.stem)
        if key and squash(result.stem).startswith(key) and (best is None or len(key) > len(best[0])):
            best = (key, ref)
    return best[1] if best else None


def solver_kinds():
    """{solver name: "GPU" or "CPU"} from the source files in src/ (.cu = CUDA GPU, .cpp = CPU)."""
    if not SRC.is_dir():
        return {}
    return {p.stem: ("GPU" if p.suffix == ".cu" else "CPU") for p in SRC.iterdir() if p.suffix in (".cpp", ".cu")}


def split_name(path, kinds):
    """(solver, model) of a final-state file named <solver>_<model>_u.vtk, or None if the solver is not one in src/."""
    stem = path.stem
    for name in sorted(kinds, key=len, reverse=True):  # longest first: ANPDJTLED_CUDA before a shorter name it starts with
        if stem.lower().startswith(name.lower() + "_") and stem.lower().endswith("_u"):
            return name, stem[len(name) + 1:-2]
    return None


def cpu_gpu_pairs(finals, algorithms):
    """[(GPU result, CPU result), ...] for the same formulation and model, and the GPU results left without a partner."""
    kinds = solver_kinds()
    family = lambda solver: solver.split("_")[0].upper()  # ANPDJTLED_CUDA and ANPDJTLED_DET_OMP are both ANPDJTLED
    named = [(path, *split_name(path, kinds)) for path in finals if split_name(path, kinds)]
    wanted = {a.lower() for a in algorithms or []}
    pairs, alone = [], []
    for gpu_path, gpu, model in named:
        if kinds[gpu] != "GPU":
            continue
        partners = [(path, cpu) for path, cpu, m in named if kinds[cpu] == "CPU" and m.lower() == model.lower() and family(cpu) == family(gpu)]
        if wanted:  # -a keeps a pair when either of its two solvers is named
            partners = [(path, cpu) for path, cpu in partners if gpu.lower() in wanted or cpu.lower() in wanted]
            if not partners and gpu.lower() not in wanted:
                continue
        if not partners:
            alone.append(f"{gpu_path.name}: no CPU result of the same formulation ({family(gpu)}) for model {model}; run run.py --final for the CPU solver too")
        pairs += [(gpu_path, path) for path, _ in partners]
    return pairs, alone, any(kinds[solver] == "GPU" for _, solver, _ in named)


def selected(path, algorithms, models):
    stem = path.stem.lower()  # <algorithm>_<model>_u
    return ((not algorithms or any(stem.startswith(a.lower() + "_") for a in algorithms)) and
            (not models or any(stem.endswith("_" + Path(m).stem.lower() + "_u") for m in models)))


def number(v):
    return f"{v:.6g}"


def percent(difference, reference):
    return "-" if reference == 0 else f"{difference / reference * 100:+.3f}%"


def print_comparison(c, tolerance, labels=("result", "reference")):
    a, b = labels  # what the two sides are called: result/reference, or GPU/CPU
    print(f"\n{c['result'].name}  vs  {c['reference'].name}   ({c['nodes']} nodes)")
    rows = [(f"Max displacement, {a}", [number(v) for v in c["max_result"]]),
            (f"Max displacement, {b}", [number(v) for v in c["max_reference"]]),
            (f"Difference ({a} - {b})", [number(v) for v in c["difference"]]),
            (f"Difference, % of {b}", [percent(d, r) for d, r in zip(c["difference"], c["max_reference"])]),
            ("Largest node-by-node difference", [number(v) for v in c["nodal"]])]
    label_width = max(len(label) for label, _ in rows)
    print("  " + " " * label_width + "".join(f"{a:>14}" for a in AXES))
    for label, values in rows:
        print("  " + label.ljust(label_width) + "".join(f"{v:>14}" for v in values))
    ok = c["relative"] * 100 <= tolerance
    print(f"  {'MATCH' if ok else 'DIFFERS'}: the largest nodal difference (at node {c['worst_node']}) is {c['relative'] * 100:.4f}% "
          f"of the {b} maximum displacement; the pass mark is {tolerance:g}%")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare the solvers' final-state VTK files with the reference results.")
    parser.add_argument("-a", "--algorithms", nargs="+", metavar="NAME", help="only results of these solvers (default: all)")
    parser.add_argument("-m", "--models", nargs="+", metavar="MODEL", help="only results for these models (default: all)")
    parser.add_argument("--results", default=str(ROOT / "results"), metavar="DIR", help="folder with the *_u.vtk files (default: results/)")
    parser.add_argument("--reference", default=str(ROOT / "reference results"), metavar="DIR", help='folder with the reference files (default: "reference results/")')
    parser.add_argument("--pair", nargs=2, action="append", metavar=("RESULT", "REFERENCE"), help="compare these two files (may be repeated); skips the automatic matching")
    parser.add_argument("--cpu-gpu", action="store_true", help="compare each GPU result with the CPU result of the same formulation and model, instead of with the reference")
    parser.add_argument("--tolerance", type=float, default=0.1, metavar="PERCENT", help="pass mark: largest nodal difference as a percentage of the reference's maximum displacement (default 0.1)")
    args = parser.parse_args()

    if args.cpu_gpu and args.pair:
        parser.error("--cpu-gpu finds its own pairs; use it without --pair")
    results_dir, reference_dir = Path(args.results), Path(args.reference)
    labels = ("GPU", "CPU") if args.cpu_gpu else ("result", "reference")
    problems = []
    if args.pair:
        pairs = [(Path(a), Path(b)) for a, b in args.pair]
        for a, b in pairs:
            for p in (a, b):
                if not p.is_file():
                    sys.exit(f"No such file: {p}")
    else:
        if not results_dir.is_dir():
            sys.exit(f"No results folder: {results_dir}. Run run.py --final first.")
        if not args.cpu_gpu and not reference_dir.is_dir():
            sys.exit(f"No reference folder: {reference_dir}")
        # with --cpu-gpu the -a filter is applied to the pairs (either solver), so both sides are loaded here
        finals = [p for p in sorted(results_dir.glob("*_u.vtk")) if selected(p, None if args.cpu_gpu else args.algorithms, args.models)]
        if not finals:
            sys.exit(f"No final-state files (*_u.vtk) to compare in {results_dir}. Run run.py --final first.")
        pairs = []
        if args.cpu_gpu:
            pairs, problems, any_gpu = cpu_gpu_pairs(finals, args.algorithms)
            if not any_gpu:
                sys.exit(f"No GPU results in {results_dir}. Run run.py --final on a machine with an NVIDIA GPU (or the Colab notebook), "
                         "then put its *_u.vtk files in this folder.")
            if not pairs and not problems:
                sys.exit(f"No GPU/CPU pair in {results_dir} involves: {', '.join(args.algorithms or [])}")
            finals = []
        for result in finals:
            reference = find_reference(result, reference_dir)
            if reference is None:
                problems.append(f"{result.name}: no matching reference in {reference_dir}")
            else:
                pairs.append((result, reference))

    comparisons, all_match = [], True
    for result, reference in pairs:
        try:
            c = compare(result, reference)
        except (OSError, ValueError) as e:
            problems.append(str(e))
            continue
        c["match"] = print_comparison(c, args.tolerance, labels)
        all_match &= c["match"]
        comparisons.append(c)

    if problems:
        print("\nNot compared:")
        for p in problems:
            print(f"  {p}")
    if comparisons:
        csv_path = (results_dir if results_dir.is_dir() else Path.cwd()) / ("comparison_cpu_gpu.csv" if args.cpu_gpu else "comparison.csv")
        a, b = (label.lower() for label in labels)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([a, b, "nodes", "quantity", f"max_{a}", f"max_{b}", "difference", "difference_percent",
                             "largest_nodal_difference", "match"])
            for c in comparisons:
                for i, axis in enumerate(AXES):
                    pct = "" if c["max_reference"][i] == 0 else f"{c['difference'][i] / c['max_reference'][i] * 100:.4f}"
                    writer.writerow([c["result"].name, c["reference"].name, c["nodes"], axis, number(c["max_result"][i]), number(c["max_reference"][i]),
                                     number(c["difference"][i]), pct, number(c["nodal"][i]), "yes" if c["match"] else "no"])
        print(f"\nSaved {csv_path}")
    return 0 if comparisons and all_match and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
