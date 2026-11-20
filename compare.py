#!/usr/bin/env python3
"""Compare the final state written by the solvers with the stored reference results.

For every final-state file (<algorithm>_<model>_u.vtk) it reports, for x, y, z and the magnitude:
  - the maximum displacement in the result and in the reference, and the difference between the two;
  - the largest node-by-node difference, which also catches fields that have the same maximum but differ elsewhere.
Values are in the units of the model file (metres for NH). "Maximum" is the largest absolute value over all nodes.

  python3 compare.py                                  every *_u.vtk in results/ against its reference
  python3 compare.py -a JANP-DJTLED_CPU                  only this solver's results
  python3 compare.py -m NH_0499 --tolerance 0.05       only this model, stricter pass mark (in %)
  python3 compare.py --pair results/JANP-DJTLED_CPU_NH_049_u.vtk "reference results/T4ANP_049_U.vtk"
  python3 compare.py --cpu-gpu                         GPU against CPU: accuracy of the final states, and computation time

The reference for a result is the file in "reference results/" named <formulation>_<tag>_U.vtk whose <tag> is the last
word of the model name: results for model NH_0499 are compared with T4ANP_0499_U.vtk, those for NH_049 with T4ANP_049_U.vtk.
ANP solvers (ANP in the name) are only matched with ANP references, and other solvers only with non-ANP references.
A result counts as a match when its largest nodal difference is within --tolerance percent (default 0.1) of the
reference's maximum displacement magnitude. The table is also saved as comparison.csv next to the results
(comparison_pairs.csv when --pair is used, so an ad-hoc comparison does not overwrite it).

With --cpu-gpu each GPU result is compared with the CPU result of the same formulation and model, e.g.
JANP-DJTLED_CUDA_NH_049_u.vtk with JANP-DJTLED_CPU_NH_049_u.vtk. A solver is a GPU one if its source in src/ is a .cu file and a
CPU one if it is a .cpp file; two solvers have the same formulation when their names start with the same word (the part
before the first underscore). That table is saved as comparison_cpu_gpu.csv. Both results must exist first:
run.py --final on a machine with an NVIDIA GPU (or the Colab notebook) writes them.

--cpu-gpu also compares computation time: it reads computation_times.csv (written by run.py --times) from the results
folder and, for each model, sets the GPU solver's time beside the CPU solver's with the speed-up (CPU time / GPU time).
Both times are from the machine that ran run.py, so run both solvers in the same run.py call. Saved as
comparison_cpu_gpu_times.csv. Either part is shown on its own when only its input exists.

Every run also writes comparison.html in the results folder: one page with everything that can be compared there
(results against the reference, GPU against CPU, computation times), whichever option was used. With --pair the page
shows those pairs and is saved as comparison_pairs.html instead. When compare.py is run from a terminal the page opens
in the browser; --no-open stops that.
Uses only the Python standard library.
"""
import argparse
import csv
import datetime
import html
import math
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # this script sits in the repository root
SRC = ROOT / "src"
AXES = ("x", "y", "z", "magnitude")


def shown(path):
    """A folder as it is named in messages and in the HTML view, without the computer-specific part of its path:
    relative to the repository when it is inside it, otherwise just its own name."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT)) or "."
    except ValueError:
        return path.name


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


def find_reference(result, reference_dir):
    """(reference file, None) for a result named <algorithm>_<model>_u.vtk, or (None, why not).
    A reference <formulation>_<tag>_U.vtk fits when <tag> is the last word of the model name and both or neither are ANP."""
    words = result.stem.split("_")  # [..., last word of the model, "u"]
    tag = words[-2].lower() if len(words) >= 3 else ""
    fits = []
    for ref in sorted(reference_dir.glob("*.vtk")):
        ref_words = ref.stem.split("_")
        if ref_words[-1].upper() == "U":
            ref_words = ref_words[:-1]
        if len(ref_words) >= 2 and ref_words[-1].lower() == tag and ("ANP" in ref.stem.upper()) == ("ANP" in result.stem.upper()):
            fits.append(ref)
    if len(fits) == 1:
        return fits[0], None
    if not fits:
        return None, f"no reference named <formulation>_{words[-2] if tag else '<tag>'}_U.vtk in {shown(reference_dir)}/"
    return None, f"more than one reference fits ({', '.join(f.name for f in fits)}); choose with --pair"


def solver_kinds():
    """{solver name: "GPU" or "CPU"} from the source files in src/ (.cu = CUDA GPU, .cpp = CPU)."""
    if not SRC.is_dir():
        return {}
    return {p.stem: ("GPU" if p.suffix == ".cu" else "CPU") for p in SRC.iterdir() if p.suffix in (".cpp", ".cu")}


def family(solver):
    """The formulation a solver belongs to: the first word of its name (JANP-DJTLED_CUDA and JANP-DJTLED_CPU are both JANP-DJTLED)."""
    return solver.split("_")[0].upper()


def split_name(path, kinds):
    """(solver, model) of a final-state file named <solver>_<model>_u.vtk, or None if the solver is not one in src/."""
    stem = path.stem
    for name in sorted(kinds, key=len, reverse=True):  # longest first: JANP-DJTLED_CUDA before a shorter name it starts with
        if stem.lower().startswith(name.lower() + "_") and stem.lower().endswith("_u"):
            return name, stem[len(name) + 1:-2]
    return None


def cpu_gpu_pairs(finals, algorithms):
    """[(GPU result, CPU result), ...] for the same formulation and model, and the GPU results left without a partner."""
    kinds = solver_kinds()
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


def time_comparison(results_dir, algorithms, models):
    """GPU against CPU computation times from computation_times.csv (run.py --times): (rows, None) or ([], why not)."""
    path = results_dir / "computation_times.csv"
    if not path.is_file():
        return [], f"no {path.name} in {results_dir}; run run.py --times with the GPU and the CPU solver"
    kinds = solver_kinds()
    with open(path, newline="") as f:
        timed = [r for r in csv.DictReader(f) if r.get("status") == "ok" and r.get("computation_ms_median")]
    wanted_solvers = {a.lower() for a in algorithms or []}
    wanted_models = {Path(m).stem.lower() for m in models or []}
    rows = []
    for gpu in timed:
        if kinds.get(gpu["algorithm"]) != "GPU" or (wanted_models and gpu["model"].lower() not in wanted_models):
            continue
        for cpu in timed:
            if kinds.get(cpu["algorithm"]) != "CPU" or cpu["model"].lower() != gpu["model"].lower() or family(cpu["algorithm"]) != family(gpu["algorithm"]):
                continue
            if wanted_solvers and gpu["algorithm"].lower() not in wanted_solvers and cpu["algorithm"].lower() not in wanted_solvers:
                continue
            cpu_ms, gpu_ms = float(cpu["computation_ms_median"]), float(gpu["computation_ms_median"])
            rows.append({"model": gpu["model"], "steps": gpu.get("steps", ""), "cpu": cpu["algorithm"], "gpu": gpu["algorithm"],
                         "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": cpu_ms / gpu_ms if gpu_ms > 0 else math.inf,
                         "runs": gpu.get("runs", ""), "cpu_threads": cpu.get("threads", "")})
    if not rows:
        return [], f"{path.name} has no model timed with both a GPU and a CPU solver; run run.py --times with both"
    return rows, None


def print_times(rows):
    print("\nComputation time, GPU against CPU (time-step loop only; from computation_times.csv)")
    headers = ["Model", "Steps", "CPU solver", "CPU (ms)", "GPU solver", "GPU (ms)", "GPU speed-up"]
    table = [[r["model"], r["steps"], r["cpu"], f"{r['cpu_ms']:,.1f}", r["gpu"], f"{r['gpu_ms']:,.1f}", f"{r['speedup']:.2f}x"] for r in rows]
    widths = [max(len(str(row[i])) for row in [headers] + table) for i in range(len(headers))]
    numeric = {1, 3, 5, 6}
    for row in [headers, ["-" * w for w in widths]] + table:
        print("  " + "  ".join(str(c).rjust(w) if i in numeric else str(c).ljust(w) for i, (c, w) in enumerate(zip(row, widths))))
    threads = sorted({r["cpu_threads"] for r in rows if r["cpu_threads"]})
    runs = sorted({r["runs"] for r in rows if r["runs"]})
    print(f"  Speed-up is CPU time / GPU time; above 1x the GPU is faster. Both are from the machine that ran run.py"
          + (f" (median of {runs[0]} runs" if len(runs) == 1 and runs[0] not in ("", "1") else " (one run" if runs == ["1"] else " (")
          + (f", CPU threads: {threads[0]})." if len(threads) == 1 else ")."))


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


# ---------------------------------------------------------------- HTML view
PAGE_STYLE = """
:root { color-scheme: light; --page: #f9f9f7; --surface: #fcfcfb; --text: #0b0b0b; --text-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --s1: #2a78d6; --s2: #eb6834; --good: #006300; --critical: #d03b3b; --code: #f0efec; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --page: #0d0d0d; --surface: #1a1a19; --text: #f0efec; --text-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --s1: #3987e5; --s2: #d95926; --good: #0ca30c; --critical: #e66767; --code: #262624; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 24px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
p { margin: 0 0 12px; }
code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--code); padding: 1px 4px; border-radius: 4px; }
.secondary { color: var(--text-2); } .muted { color: var(--muted); }
.note { color: var(--text-2); font-size: 13px; margin: 12px 0 0; }
header { margin-bottom: 20px; }
.card, .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
.card { padding: 20px; margin-bottom: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { padding: 16px; }
.tile-label { color: var(--text-2); font-size: 13px; }
.tile-value { font-size: 26px; font-weight: 600; margin: 2px 0; }
.tile-sub { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 12px; color: var(--text-2); font-size: 13px; }
.key { display: inline-flex; align-items: center; gap: 6px; }
.key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.key i.line { width: 1px; height: 14px; border-radius: 0; background: var(--muted); }
.bars { display: flex; flex-direction: column; gap: 6px; margin: 8px 0 4px; }
.bar-group { font-size: 13px; font-weight: 600; margin: 10px 0 2px; display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
.bar-group:first-child { margin-top: 0; }
.bar-group span { font-weight: 400; color: var(--text-2); }
.bar-row { display: flex; align-items: center; gap: 12px; border-radius: 6px; outline-offset: 2px; }
.bar-row:hover .bar, .bar-row:focus .bar { filter: brightness(1.08); }
.bar-label { width: 230px; flex: none; font-size: 13px; text-align: right; overflow-wrap: anywhere; }
.bar-track { flex: 1; position: relative; display: flex; align-items: center; gap: 8px; min-width: 0; height: 28px; border-left: 1px solid var(--axis); }
.bar { height: 18px; border-radius: 0 4px 4px 0; min-width: 2px; background: var(--s1); }
.bar.gpu { background: var(--s2); }
.bar-val { font-size: 13px; color: var(--text-2); white-space: nowrap; font-variant-numeric: tabular-nums; background: var(--surface); padding: 0 2px; position: relative; z-index: 1; }
.mark { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--muted); }
.bar-status { width: 84px; flex: none; }
.status { display: inline-flex; gap: 4px; align-items: center; font-weight: 600; white-space: nowrap; font-size: 13px; }
.status.good { color: var(--good); } .status.critical { color: var(--critical); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; font-weight: 600; color: var(--text-2); border-bottom: 1px solid var(--axis); padding: 6px 10px; white-space: nowrap; }
td { border-bottom: 1px solid var(--grid); padding: 6px 10px; vertical-align: top; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
details { margin-top: 12px; } summary { cursor: pointer; color: var(--text-2); padding: 4px 0; }
details > .table-wrap { margin: 6px 0 14px; }
ul.plain { margin: 6px 0 0; padding-left: 18px; color: var(--text-2); }
#tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--text); border: 1px solid var(--border);
  box-shadow: 0 4px 16px rgba(0,0,0,0.15); border-radius: 8px; padding: 6px 10px; font-size: 12px; white-space: pre-line; max-width: 300px; }
footer { font-size: 12px; margin-top: 24px; }
@media (max-width: 640px) { .bar-label { width: 120px; } .bar-status { width: auto; } }
"""

PAGE_SCRIPT = """
(function () {
  const tip = document.getElementById("tip");
  function show(text, x, y) {
    tip.textContent = text; tip.hidden = false;
    tip.style.left = Math.max(8, Math.min(x + 14, innerWidth - tip.offsetWidth - 8)) + "px";
    tip.style.top = Math.max(8, Math.min(y + 14, innerHeight - tip.offsetHeight - 8)) + "px";
  }
  document.querySelectorAll("[data-tip]").forEach(function (el) {
    el.addEventListener("mousemove", function (e) { show(el.dataset.tip, e.clientX, e.clientY); });
    el.addEventListener("mouseleave", function () { tip.hidden = true; });
    el.addEventListener("focus", function () { const r = el.getBoundingClientRect(); show(el.dataset.tip, r.left + 240, r.bottom - 6); });
    el.addEventListener("blur", function () { tip.hidden = true; });
  });
})();
"""


def esc(text):
    return html.escape(str(text), quote=True)


def html_table(headers, rows, numeric=()):
    head = "".join(f'<th{" class=num" if i in numeric else ""}>{h}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td{" class=num" if i in numeric else ""}>{cell}</td>' for i, cell in enumerate(row)) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def status_badge(ok):
    return (f'<span class="status {"good" if ok else "critical"}"><span aria-hidden="true">{"&#10003;" if ok else "&#10005;"}</span> '
            f'{"Match" if ok else "Differs"}</span>')


def result_label(path, kinds):
    """'solver, model' for a result file named <solver>_<model>_u.vtk, else the file name."""
    parts = split_name(path, kinds)
    return f"{parts[0]} · {parts[1]}" if parts else path.name


def accuracy_section(title, lead, labels, comparisons, problems, tolerance, kinds):
    """One card: a bar per comparison (its largest nodal difference against the pass mark), then the numbers."""
    if not comparisons and not problems:
        return ""
    a, b = labels
    out = [f'<section class="card"><h2>{esc(title)}</h2><p class="secondary">{lead}</p>']
    if comparisons:
        finite = [c["relative"] * 100 for c in comparisons if math.isfinite(c["relative"])]
        top = max([tolerance] + finite) * 1.2  # the bar scale: room for the largest value and the pass mark
        out.append('<div class="legend"><span class="key"><i style="background:var(--s1)"></i>Largest node-by-node difference, % of the '
                   f'{esc(b)} maximum displacement</span><span class="key"><i class="line"></i>Pass mark, {tolerance:g}%</span></div><div class="bars">')
        for c in comparisons:
            rel = c["relative"] * 100
            shown = f"{rel:.4f}%" if math.isfinite(rel) else "–"
            tip = (f"{c['result'].name}\nagainst {c['reference'].name}\nLargest nodal difference: {number(c['nodal'][3])} ({shown} of the maximum)\n"
                   f"at node {c['worst_node']} of {c['nodes']}\nPass mark: {tolerance:g}%")
            width = min(1.0, rel / top) if math.isfinite(rel) else 1.0
            out.append(f'<div class="bar-row" tabindex="0" data-tip="{esc(tip)}"><div class="bar-label">{esc(result_label(c["result"], kinds))}</div>'
                       f'<div class="bar-track"><div class="mark" style="left:calc((100% - 84px) * {tolerance / top:.4f})"></div>'
                       f'<div class="bar" style="width:calc((100% - 84px) * {width:.4f})"></div><span class="bar-val">{shown}</span></div>'
                       f'<div class="bar-status">{status_badge(c["match"])}</div></div>')
        out.append("</div>")
        rows = [[f"<b>{esc(result_label(c['result'], kinds))}</b>", f"<code>{esc(c['reference'].name)}</code>", c["nodes"],
                 number(c["max_result"][3]), number(c["max_reference"][3]), number(c["nodal"][3]),
                 f"{c['relative'] * 100:.4f}%" if math.isfinite(c["relative"]) else "–", status_badge(c["match"])] for c in comparisons]
        out.append(html_table([esc(a.capitalize()), esc(b.capitalize()), "Nodes", f"Max |u|, {esc(a)}", f"Max |u|, {esc(b)}",
                               "Largest nodal difference", "% of max |u|", "Status"], rows, numeric=(2, 3, 4, 5, 6)))
        out.append("<details><summary>Maximum displacement in x, y, z and magnitude</summary>")
        for c in comparisons:
            detail = [[f"Max displacement, {esc(a)}"] + [number(v) for v in c["max_result"]],
                      [f"Max displacement, {esc(b)}"] + [number(v) for v in c["max_reference"]],
                      [f"Difference ({esc(a)} − {esc(b)})"] + [number(v) for v in c["difference"]],
                      [f"Difference, % of {esc(b)}"] + [percent(d, r) for d, r in zip(c["difference"], c["max_reference"])],
                      ["Largest node-by-node difference"] + [number(v) for v in c["nodal"]]]
            out.append(f'<p class="secondary" style="margin:10px 0 0"><code>{esc(c["result"].name)}</code> against <code>{esc(c["reference"].name)}</code></p>'
                       + html_table([""] + list(AXES), detail, numeric=(1, 2, 3, 4)))
        out.append("</details>")
    if problems:
        out.append('<p class="note">Not compared:</p><ul class="plain">' + "".join(f"<li>{esc(p)}</li>" for p in problems) + "</ul>")
    out.append("</section>")
    return "".join(out)


def times_section(times):
    if not times:
        return ""
    top = max(max(r["cpu_ms"], r["gpu_ms"]) for r in times)
    out = ['<section class="card"><h2>GPU against CPU: computation time</h2>'
           '<p class="secondary">Time in the time-step loop, as timed by <code>run.py --times</code>; writing VTK files is not included. Lower is faster.</p>'
           '<div class="legend"><span class="key"><i style="background:var(--s1)"></i>CPU</span>'
           '<span class="key"><i style="background:var(--s2)"></i>GPU</span></div><div class="bars">']
    for r in times:
        out.append(f'<div class="bar-group">{esc(r["model"])}<span>GPU {r["speedup"]:.1f}× {"faster" if r["speedup"] >= 1 else "the speed"}</span></div>')
        for kind, solver, ms in (("cpu", r["cpu"], r["cpu_ms"]), ("gpu", r["gpu"], r["gpu_ms"])):
            tip = f"{solver}\n{r['model']}, {r['steps']} steps\n{ms:,.1f} ms" + (f"\n{ms / float(r['steps']) * 1000:.1f} microseconds per step" if str(r["steps"]).isdigit() else "")
            out.append(f'<div class="bar-row" tabindex="0" data-tip="{esc(tip)}"><div class="bar-label">{esc(solver)}</div><div class="bar-track">'
                       f'<div class="bar {kind}" style="width:calc((100% - 96px) * {ms / top:.4f})"></div><span class="bar-val">{ms:,.1f} ms</span></div></div>')
    out.append("</div>")
    out.append(html_table(["Model", "Steps", "CPU solver", "CPU (ms)", "GPU solver", "GPU (ms)", "GPU speed-up"],
                          [[f"<b>{esc(r['model'])}</b>", esc(r["steps"]), esc(r["cpu"]), f"{r['cpu_ms']:,.1f}", esc(r["gpu"]), f"{r['gpu_ms']:,.1f}", f"{r['speedup']:.2f}×"]
                           for r in times], numeric=(1, 3, 5, 6)))
    runs = sorted({r["runs"] for r in times if r["runs"]})
    threads = sorted({r["cpu_threads"] for r in times if r["cpu_threads"]})
    out.append('<p class="note">Speed-up is CPU time divided by GPU time. Both times come from the machine that ran <code>run.py</code>'
               + (f", median of {esc(runs[0])} runs" if len(runs) == 1 and runs[0] != "1" else "")
               + (f", CPU threads: {esc(threads[0])}" if len(threads) == 1 else "") + ".</p></section>")
    return "".join(out)


def run_comparisons(pairs, tolerance):
    """compare() every pair: (comparisons with their match flag, problems)."""
    done, problems = [], []
    for result, reference in pairs:
        try:
            c = compare(result, reference)
        except (OSError, ValueError) as e:
            problems.append(str(e))
            continue
        c["match"] = c["relative"] * 100 <= tolerance
        done.append(c)
    return done, problems


def write_html(args, results_dir, reference_dir, pair_comparisons):
    """comparison.html: everything that can be compared in the results folder, or the --pair comparisons."""
    kinds = solver_kinds()
    sections, tiles, times = [], [], []
    if pair_comparisons is not None:
        sections.append(accuracy_section("Selected pairs", "The files given with <code>--pair</code>.", ("result", "reference"), pair_comparisons, [], args.tolerance, kinds))
        if pair_comparisons:
            tiles.append(("Pairs that match", f"{sum(c['match'] for c in pair_comparisons)} of {len(pair_comparisons)}", f"pass mark {args.tolerance:g}%"))
    else:
        finals = [p for p in sorted(results_dir.glob("*_u.vtk")) if selected(p, None, args.models)]
        if reference_dir.is_dir():
            pairs, problems = [], []
            for result in (p for p in finals if selected(p, args.algorithms, None)):
                reference, why_not = find_reference(result, reference_dir)
                if reference is None:
                    problems.append(f"{result.name}: {why_not}")
                else:
                    pairs.append((result, reference))
            done, more = run_comparisons(pairs, args.tolerance)
            sections.append(accuracy_section("Accuracy against the reference",
                                             "Each solver result's final displacements against the reference result for the same model.",
                                             ("result", "reference"), done, problems + more, args.tolerance, kinds))
            if done:
                tiles.append(("Results that match the reference", f"{sum(c['match'] for c in done)} of {len(done)}", f"pass mark {args.tolerance:g}% of the maximum displacement"))
        pairs, alone, _ = cpu_gpu_pairs(finals, args.algorithms)
        done, more = run_comparisons(pairs, args.tolerance)
        sections.append(accuracy_section("GPU against CPU: accuracy", "Each GPU result against the CPU result of the same formulation and model.",
                                         ("GPU", "CPU"), done, alone + more, args.tolerance, kinds))
        if done:
            tiles.append(("GPU results that match the CPU", f"{sum(c['match'] for c in done)} of {len(done)}", f"largest difference {max(c['relative'] for c in done) * 100:.4f}%"))
        times, _ = time_comparison(results_dir, args.algorithms, args.models)
        if times:
            low, high = min(r["speedup"] for r in times), max(r["speedup"] for r in times)
            tiles.append(("GPU speed-up over the CPU", f"{low:.1f}×" if high - low < 0.05 else f"{low:.1f}–{high:.1f}×",
                          f"{times[0]['gpu']} vs {times[0]['cpu']}, same machine"))
        sections.append(times_section(times))
    body = "".join(sections) or '<section class="card"><p class="secondary">Nothing to compare yet. Run <code>run.py --final --times</code> first.</p></section>'
    page = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>JANP-DJTLED comparison</title><style>{PAGE_STYLE}</style></head><body><main>",
            f'<header><h1>JANP-DJTLED comparison</h1><p class="secondary">Results in <code>{esc(shown(results_dir))}/</code>. '
            "Displacements are in the model's own units (metres for the NH models). A result matches when its largest node-by-node "
            f"difference is within {args.tolerance:g}% of the maximum displacement it is compared with.</p></header>"]
    if tiles:
        page.append('<section class="tiles">' + "".join(f'<div class="tile"><div class="tile-label">{esc(label)}</div><div class="tile-value">{esc(value)}</div>'
                                                          f'<div class="tile-sub">{esc(sub)}</div></div>' for label, value, sub in tiles) + "</section>")
    page.append(body)
    page.append(f'<footer class="muted">Written by <code>compare.py</code> on {datetime.datetime.now():%Y-%m-%d %H:%M}.</footer>'
                f'</main><div id="tip" hidden></div><script>{PAGE_SCRIPT}</script></body></html>\n')
    name = "comparison_pairs.html" if pair_comparisons is not None else "comparison.html"  # an ad-hoc --pair run does not replace the full page
    path = (results_dir if results_dir.is_dir() else Path.cwd()) / name
    path.write_text("".join(page), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Compare the solvers' final-state VTK files with the reference results.")
    parser.add_argument("-a", "--algorithms", nargs="+", metavar="NAME", help="only results of these solvers (default: all)")
    parser.add_argument("-m", "--models", nargs="+", metavar="MODEL", help="only results for these models (default: all)")
    parser.add_argument("--results", default=str(ROOT / "results"), metavar="DIR", help="folder with the *_u.vtk files (default: results/)")
    parser.add_argument("--reference", default=str(ROOT / "reference results"), metavar="DIR", help='folder with the reference files (default: "reference results/")')
    parser.add_argument("--pair", nargs=2, action="append", metavar=("RESULT", "REFERENCE"), help="compare these two files (may be repeated); skips the automatic matching")
    parser.add_argument("--cpu-gpu", action="store_true", help="GPU against CPU for the same formulation and model: accuracy of the final states, and computation time with the speed-up")
    parser.add_argument("--tolerance", type=float, default=0.1, metavar="PERCENT", help="pass mark: largest nodal difference as a percentage of the reference's maximum displacement (default 0.1)")
    parser.add_argument("--no-open", action="store_true", help="do not open the HTML view in the browser (it is still written)")
    args = parser.parse_args()

    if args.cpu_gpu and args.pair:
        parser.error("--cpu-gpu finds its own pairs; use it without --pair")
    results_dir, reference_dir = Path(args.results), Path(args.reference)
    labels = ("GPU", "CPU") if args.cpu_gpu else ("result", "reference")
    problems, times, notes = [], [], []
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
        if not finals and not args.cpu_gpu:
            sys.exit(f"No final-state files (*_u.vtk) to compare in {results_dir}. Run run.py --final first.")
        pairs = []
        if args.cpu_gpu:
            times, no_times = time_comparison(results_dir, args.algorithms, args.models)
            pairs, problems, any_gpu = cpu_gpu_pairs(finals, args.algorithms)
            if not any_gpu and not times:
                sys.exit(f"No GPU results in {results_dir}. Run run.py --final --times on a machine with an NVIDIA GPU (or the Colab notebook), "
                         "then put its results in this folder.")
            if not any_gpu:
                notes.append("Accuracy not compared: no GPU final-state files (*_u.vtk) here; run run.py --final for them.")
            elif not pairs and not problems and not times:
                sys.exit(f"No GPU/CPU pair in {results_dir} involves: {', '.join(args.algorithms or [])}")
            if no_times:
                notes.append(f"Computation time not compared: {no_times}.")
            finals = []
        for result in finals:
            reference, why_not = find_reference(result, reference_dir)
            if reference is None:
                problems.append(f"{result.name}: {why_not}")
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
        csv_path = (results_dir if results_dir.is_dir() else Path.cwd()) / ("comparison_cpu_gpu.csv" if args.cpu_gpu else "comparison_pairs.csv" if args.pair else "comparison.csv")
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
    if times:
        print_times(times)
        times_path = results_dir / "comparison_cpu_gpu_times.csv"
        with open(times_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "steps", "cpu_solver", "cpu_ms", "gpu_solver", "gpu_ms", "gpu_speedup", "runs", "cpu_threads"])
            for r in times:
                writer.writerow([r["model"], r["steps"], r["cpu"], f"{r['cpu_ms']:.3f}", r["gpu"], f"{r['gpu_ms']:.3f}", f"{r['speedup']:.3f}", r["runs"], r["cpu_threads"]])
        print(f"\nSaved {times_path}")
    for note in notes:
        print(f"\n{note}")
    html_path = write_html(args, results_dir, reference_dir, comparisons if args.pair else None)
    print(f"\nHTML view: {html_path}")
    if not args.no_open and sys.stdout.isatty():  # only when a person runs it in a terminal, not in a script or notebook
        webbrowser.open(html_path.resolve().as_uri())
    return 0 if (comparisons or times) and all_match and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
