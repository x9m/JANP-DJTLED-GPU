#!/usr/bin/env python3
"""Build and run selected solvers on selected model files, and keep only the outputs you ask for.

Run from anywhere; paths are relative to the repository root.

  python3 run.py --list                         show the solvers and models that can be selected
  python3 run.py                                every solver on every model, all three outputs
  python3 run.py -a DJTLED -m NH --times        one solver, one model, computation time only
  python3 run.py -a DJTLED ANPDJTLED_DET_OMP --final --animation --frames 50
  python3 run.py --times --repeat 5 --threads 4

Selecting what to run
  -a / --algorithms   solver names: the source files in src/ without their extension (default: all)
  -m / --models       model names from models/ (with or without .txt) or paths to model files (default: all)

Selecting the outputs (any combination; with none of the three given, all three are produced)
  --animation   <algorithm>_<model>_simulation.vtk.series and the <algorithm>_<model>_timesteps/ folder
  --final       <algorithm>_<model>_u.vtk and <algorithm>_<model>_undeformed.vtk
  --times       a table of computation times, also saved as computation_times.csv

Outputs go to results/ (change with --out). A solver is compiled into bin/ first if its program is missing or older
than its source, so an edited source is never run from a stale program. Uses only the Python standard library.
"""
import argparse
import csv
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # this script sits in the repository root
SRC, BIN, MODELS = ROOT / "src", ROOT / "bin", ROOT / "models"
SHARED_HEADER = SRC / "vtk_output.h"  # included by every solver, so a change to it also means a rebuild


def find_solvers():
    """{name: source file} for every solver in src/ (.cpp = OpenMP on the CPU, .cu = CUDA on the GPU)."""
    return {p.stem: p for p in sorted(SRC.iterdir()) if p.suffix in (".cpp", ".cu")}


def find_models():
    return {p.stem: p for p in sorted(MODELS.glob("*.txt"))}


def choose_solvers(wanted, solvers):
    if not wanted:
        return list(solvers)
    by_lower = {name.lower(): name for name in solvers}
    unknown = [w for w in wanted if w.lower() not in by_lower]
    if unknown:
        sys.exit(f"Unknown algorithm: {', '.join(unknown)}. Available: {', '.join(solvers)}")
    return list(dict.fromkeys(by_lower[w.lower()] for w in wanted))  # keep the order given, drop repeats


def choose_models(wanted, models):
    if not wanted:
        return list(models.values())
    by_lower = {name.lower(): path for name, path in models.items()}
    chosen = []
    for w in wanted:
        if Path(w).is_file():  # a path to a model file anywhere
            chosen.append(Path(w).resolve())
        elif Path(w).stem.lower() in by_lower:  # a name in models/, with or without .txt
            chosen.append(by_lower[Path(w).stem.lower()])
        else:
            sys.exit(f"Unknown model: {w}. Available in models/: {', '.join(models) or 'none'} (or give a path to a file)")
    return list(dict.fromkeys(chosen))


def build_command(source, program):
    """The compiler command for a solver, or (None, reason) when this machine cannot build it."""
    if source.suffix == ".cu":
        if not shutil.which("nvcc"):
            return None, "no CUDA compiler (nvcc) on this machine"
        return ["nvcc", "-std=c++17", "-O3", str(source), "-o", str(program)], None
    if platform.system() == "Darwin":  # Apple clang has no OpenMP of its own; use Homebrew's libomp
        candidates = (os.environ.get("LIBOMP_PREFIX"), "/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")
        omp = next((p for p in candidates if p and Path(p, "include", "omp.h").is_file()), None)
        if not omp:
            return None, "OpenMP not found; install it with: brew install libomp"
        return ["clang++", "-std=c++17", "-O3", "-Xclang", "-fopenmp", f"-I{omp}/include", str(source),
                f"-L{omp}/lib", "-lomp", "-o", str(program)], None
    return ["g++", "-std=c++17", "-O3", "-fopenmp", str(source), "-o", str(program)], None


def build(name, source):
    """Compile the solver if its program is missing or older than its source. Returns (program, problem)."""
    program = BIN / name
    newest_source = max(p.stat().st_mtime for p in (source, SHARED_HEADER) if p.exists())
    if program.exists() and program.stat().st_mtime >= newest_source:
        return program, None
    command, problem = build_command(source, program)
    if command is None:
        return None, problem
    print(f"  building {name} ...", flush=True)
    BIN.mkdir(exist_ok=True)
    try:
        done = subprocess.run(command, capture_output=True, text=True)
    except OSError as e:
        return None, f"cannot run {command[0]}: {e}"
    if done.returncode != 0:
        errors = [line for line in done.stderr.splitlines() if "error" in line][:5]
        return None, "build failed:\n      " + "\n      ".join(errors or done.stderr.splitlines()[-5:])
    return program, None


def number_after(label, text):
    m = re.search(re.escape(label) + r"\s*([0-9.eE+-]+)", text)
    return float(m.group(1)) if m else None


def run_once(program, model, frames, workdir, threads):
    """One run in workdir. Returns (ok, computation ms, export ms, number of steps, error text)."""
    env = dict(os.environ)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    try:
        done = subprocess.run([str(program), str(model), str(frames)], cwd=workdir, capture_output=True, text=True, env=env)
    except OSError as e:  # e.g. a program in bin/ that was built on another system (a macOS program copied to Linux)
        return False, None, None, None, f"cannot start {program.name}: {e.strerror}; if bin/ came from another computer, delete bin/ and run again"
    out = done.stdout
    steps = number_after("NumSteps:", out)
    if done.returncode != 0:
        errors = [line.strip() for line in (out + done.stderr).splitlines() if "Error" in line]
        return False, None, None, steps, errors[-1] if errors else f"exit code {done.returncode}"
    return True, number_after("Computation time:", out), number_after("Export time:", out), steps, None


def keep(workdir, out_dir, names):
    """Move the named files/folders from the run folder into the output folder, replacing older ones of the same name."""
    kept = []
    for name in names:
        source, target = workdir / name, out_dir / name
        if not source.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        kept.append(name + ("/" if target.is_dir() else ""))
    return kept


def print_table(rows, repeat):
    headers = ["Algorithm", "Model", "Status", "Steps", "Computation (ms)" + (f", median of {repeat}" if repeat > 1 else "")]
    if repeat > 1:
        headers.append("Best (ms)")
    headers.append("VTK export (ms)")
    fmt = lambda v: "-" if v is None else f"{v:,.1f}"
    table = []
    for r in rows:
        line = [r["algorithm"], r["model"], r["status"], "-" if r["steps"] is None else f"{r['steps']:.0f}", fmt(r["median_ms"])]
        if repeat > 1:
            line.append(fmt(r["best_ms"]))
        line.append(fmt(r["export_ms"]))
        table.append(line)
    widths = [max(len(str(row[i])) for row in [headers] + table) for i in range(len(headers))]
    numeric = set(range(3, len(headers)))
    for row in [headers, ["-" * w for w in widths]] + table:
        print("  " + "  ".join(str(c).rjust(w) if i in numeric else str(c).ljust(w) for i, (c, w) in enumerate(zip(row, widths))))


def main():
    parser = argparse.ArgumentParser(description="Build and run selected solvers on selected models.",
                                     epilog="With none of --animation, --final and --times, all three are produced.")
    parser.add_argument("-a", "--algorithms", nargs="+", metavar="NAME", help="solvers to run (default: all in src/)")
    parser.add_argument("-m", "--models", nargs="+", metavar="MODEL", help="model names in models/ or paths (default: all in models/)")
    parser.add_argument("--animation", action="store_true", help="keep the animation (.vtk.series and the timesteps folder)")
    parser.add_argument("--final", action="store_true", help="keep the final state (_u.vtk and _undeformed.vtk)")
    parser.add_argument("--times", action="store_true", help="report the computation times (table and computation_times.csv)")
    parser.add_argument("--frames", type=int, default=100, metavar="N", help="animation frames to save (default 100)")
    parser.add_argument("--threads", type=int, metavar="N", help="OpenMP threads (default: the solver's own choice, all cores)")
    parser.add_argument("--repeat", type=int, default=1, metavar="N", help="run each case N times and report the median time (default 1)")
    parser.add_argument("--out", default=str(ROOT / "results"), metavar="DIR", help="output folder (default: results/)")
    parser.add_argument("--list", action="store_true", help="list the available algorithms and models, then stop")
    args = parser.parse_args()

    solvers, models = find_solvers(), find_models()
    if args.list:
        print("Algorithms (src/):")
        for name, source in solvers.items():
            _, problem = build_command(source, BIN / name)
            print(f"  {name:<22} {'CUDA GPU' if source.suffix == '.cu' else 'OpenMP CPU'}{'   [cannot build here: ' + problem + ']' if problem else ''}")
        print("Models (models/):")
        for name, path in models.items():
            print(f"  {name:<22} {path.relative_to(ROOT)}")
        return 0
    if args.frames < 1 or args.repeat < 1 or (args.threads is not None and args.threads < 1):
        parser.error("--frames, --repeat and --threads must be 1 or more")

    if not (args.animation or args.final or args.times):
        args.animation = args.final = args.times = True
    chosen_solvers, chosen_models = choose_solvers(args.algorithms, solvers), choose_models(args.models, models)
    if not chosen_solvers or not chosen_models:
        sys.exit("Nothing to run: no solvers in src/ or no models in models/.")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = args.frames if args.animation else 0  # 0 = the solver writes no animation frames
    wanted = [w for w, on in (("animation", args.animation), ("final state", args.final), ("computation times", args.times)) if on]
    print(f"Outputs: {', '.join(wanted)}   ->   {out_dir}")

    rows = []
    for name in chosen_solvers:
        print(f"{name}")
        program, problem = build(name, solvers[name])
        for model in chosen_models:
            row = {"algorithm": name, "model": model.stem, "status": "ok", "steps": None, "runs": 0,
                   "median_ms": None, "best_ms": None, "export_ms": None}
            rows.append(row)
            if program is None:
                row["status"] = "skipped" if "on this machine" in problem or "not found" in problem else "build failed"
                print(f"  {model.stem}: {row['status']} ({problem})")
                continue
            times, error = [], None
            # each run happens in its own empty folder, so only the outputs that were asked for reach the output folder
            with tempfile.TemporaryDirectory(dir=out_dir, prefix=".run_") as tmp:
                for _ in range(args.repeat):
                    ok, compute_ms, export_ms, steps, error = run_once(program, model, frames, tmp, args.threads)
                    row["steps"] = steps
                    if not ok:
                        break
                    row["runs"] += 1
                    row["export_ms"] = export_ms
                    if compute_ms is not None:
                        times.append(compute_ms)
                if error:
                    row["status"] = "failed"
                    print(f"  {model.stem}: failed ({error})")
                    continue
                prefix = f"{name}_{model.stem}"  # the solvers' own naming rule: <algorithm>_<model>
                kept = []
                if args.final:
                    kept += keep(Path(tmp), out_dir, [f"{prefix}_u.vtk", f"{prefix}_undeformed.vtk"])
                if args.animation:
                    kept += keep(Path(tmp), out_dir, [f"{prefix}_simulation.vtk.series", f"{prefix}_timesteps"])
            if times:
                row["median_ms"], row["best_ms"] = statistics.median(times), min(times)
            took = f"{row['median_ms']:,.1f} ms" if times else "time not reported"
            print(f"  {model.stem}: ok, {took}" + (f"; saved {', '.join(kept)}" if kept else ""))

    if args.times:
        print("\nComputation time (time-step loop only; writing VTK files is timed separately)")
        print_table(rows, args.repeat)
        csv_path = out_dir / "computation_times.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["algorithm", "model", "status", "steps", "runs", "threads", "computation_ms_median", "computation_ms_best", "vtk_export_ms"])
            for r in rows:
                writer.writerow([r["algorithm"], r["model"], r["status"], "" if r["steps"] is None else int(r["steps"]), r["runs"],
                                 args.threads or "default"] + ["" if r[k] is None else f"{r[k]:.3f}" for k in ("median_ms", "best_ms", "export_ms")])
        print(f"\n  saved {csv_path}")
    return 1 if any(r["status"] in ("failed", "build failed") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
