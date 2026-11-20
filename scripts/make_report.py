#!/usr/bin/env python3
"""Build an HTML report from the output of src/cpu/run_all_cpu.sh and src/gpu/run_all_gpu.sh.

Usage:  python3 scripts/make_report.py [results_dir]      (default: results)

Reads <results_dir>/run_summary_cpu.tsv and/or run_summary_gpu.tsv (written by the two run scripts; whichever exist),
each program's log.txt and VTK files,
and writes <results_dir>/report.html: a single self-contained page that works offline.
Uses only the Python standard library.
"""
import array
import base64
import datetime
import html
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys

# Pairs of programs whose final displacements are compared.
# expect_match=True: same formulation, different implementation, so results should agree.
# expect_match=False: the formulation differs on purpose (ANP vs standard), so the difference is the effect being studied.
PAIRS = [
    # (program A,  program B,        expect_match, what the comparison shows)
    ("TLED",       "TLED_OMP",       True,  "Serial vs OpenMP"),
    ("ANPTLED",    "ANPTLED_OMP",    True,  "Serial vs OpenMP, with ANP"),
    ("TLED",       "DJTLED_OMP",     True,  "TLED vs direct Jacobian TLED"),
    ("ANPTLED",    "ANPDJTLED_OMP",  True,  "TLED vs direct Jacobian TLED, with ANP"),
    ("TLED",       "ANPTLED",        False, "Effect of ANP on TLED"),
    ("DJTLED_OMP", "ANPDJTLED_OMP",  False, "Effect of ANP on DJTLED"),
    ("ANPDJTLED_OMP", "ANPDJTLED_DET_OMP",  True, "ANP from nodal pressures vs from Jacobian determinants"),
    ("ANPDJTLED_OMP", "ANPDJTLED_FAST_OMP", True, "ANP, original vs fewer operations per element"),
    # GPU (CUDA) versions: each should match its CPU version
    ("TLED",          "TLED_CUDA",      True,  "CPU vs CUDA GPU"),
    ("ANPTLED",       "ANPTLED_CUDA",   True,  "CPU vs CUDA GPU, with ANP"),
    ("DJTLED_OMP",    "DJTLED_CUDA",    True,  "CPU vs CUDA GPU, direct Jacobian"),
    ("ANPDJTLED_OMP", "ANPDJTLED_CUDA", True,  "CPU vs CUDA GPU, direct Jacobian with ANP"),
    ("TLED_CUDA",     "ANPTLED_CUDA",   False, "Effect of ANP on TLED, GPU"),
    ("DJTLED_CUDA",   "ANPDJTLED_CUDA", False, "Effect of ANP on DJTLED, GPU"),
]
MATCH_TOL = 1e-3   # two runs "match" if their largest nodal difference is within 0.1% of the largest displacement
BASELINE  = "TLED" # speed-ups are relative to this program
LOG_KEYS  = {"Model", "Nodes", "Elements", "Formulation", "EleMaterial", "BC", "DampingCoef.",
             "TimeStep", "TotalTime", "NumSteps", "Threads", "Computation time", "GPU", "Export time"}
# CPU vs GPU speed comparison: (algorithm, serial CPU program, OpenMP CPU program, GPU program)
SPEED_GROUPS = [
    ("TLED",      "TLED",    "TLED_OMP",      "TLED_CUDA"),
    ("ANPTLED",   "ANPTLED", "ANPTLED_OMP",   "ANPTLED_CUDA"),
    ("DJTLED",    None,      "DJTLED_OMP",    "DJTLED_CUDA"),
    ("ANPDJTLED", None,      "ANPDJTLED_OMP", "ANPDJTLED_CUDA"),
]


# ---------------------------------------------------------------- reading run output
def to_float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


RUN_KINDS = ("cpu", "gpu")  # run_summary_<kind>.tsv, written by src/<kind>/run_all_<kind>.sh


def read_summaries(results):
    """Every run_summary_<kind>.tsv that exists: {kind: meta}, and all program rows (CPU first)."""
    metas, rows = {}, []
    for kind in RUN_KINDS:
        path = os.path.join(results, f"run_summary_{kind}.tsv")
        if os.path.isfile(path):
            metas[kind], kind_rows = read_summary(path)
            rows += [dict(r, kind=kind) for r in kind_rows]
    return metas, rows


def read_summary(path):
    """One run summary: '# key=value' lines, then a header row and one tab-separated row per program."""
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].partition("=")
                meta[key.strip()] = value.strip()
                continue
            cols = (line.split("\t") + [""] * 5)[:5]
            if cols[0] == "name":
                continue
            rows.append({"name": cols[0], "status": cols[1], "time_ms": to_float(cols[2]),
                         "model": cols[3], "openmp": cols[4] == "yes"})
    return meta, rows


def parse_log(path):
    """Key/value lines such as '\tNodes:\t\t1695 (5085 DOFs)' from a program's console log."""
    info, lines = {}, []
    if os.path.isfile(path):
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    for line in lines:
        m = re.match(r"\s*([A-Za-z. ]+?):\t+(.*\S)\s*$", line)
        if m and m.group(1) in LOG_KEYS:
            info.setdefault(m.group(1), m.group(2))
        m = re.match(r"\s*using (\d+) threads", line)
        if m:
            info.setdefault("Threads", m.group(1))
    if "Threads" in info:
        m = re.match(r"\d+", info["Threads"])
        info["Threads"] = m.group(0) if m else info["Threads"]
    return info, lines


def read_vtk_vectors(path):
    """Displacement vectors from the POINT_DATA / VECTORS block of a legacy ASCII VTK file."""
    try:
        with open(path) as f:
            lines = f.read().split("\n")
        n = None
        for i, line in enumerate(lines):
            if line.startswith("POINT_DATA"):
                n = int(line.split()[1])
            elif line.startswith("VECTORS") and n is not None:
                return [tuple(float(v) for v in lines[j].split()) for j in range(i + 1, i + 1 + n)]
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_vtk_mesh(path):
    """Node coordinates (POINTS) and 4-node tetrahedra (CELLS) from a legacy ASCII VTK file."""
    pts, tets = None, None
    try:
        with open(path) as f:
            lines = f.read().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("POINTS"):
                n = int(line.split()[1])
                pts = [tuple(float(v) for v in lines[j].split()) for j in range(i + 1, i + 1 + n)]
            elif line.startswith("CELLS"):
                n = int(line.split()[1])
                tets = [tuple(int(v) for v in lines[j].split()[1:5]) for j in range(i + 1, i + 1 + n)]
    except (OSError, ValueError, IndexError):
        return None, None
    return pts, tets


def surface_mesh(pts, tets):
    """Outer surface of a tetrahedral mesh: faces that belong to only one tet, wound to face outwards.
    Returns (surface node ids, triangle corners as indices into that node list)."""
    faces = {}
    for t in tets:
        for face, opposite in (((t[0], t[1], t[2]), t[3]), ((t[0], t[1], t[3]), t[2]),
                               ((t[0], t[2], t[3]), t[1]), ((t[1], t[2], t[3]), t[0])):
            key = tuple(sorted(face))
            faces[key] = None if key in faces else (face, opposite)  # a face shared by two tets is interior
    tris = []
    for entry in faces.values():
        if entry is None:
            continue
        (a, b, c), d = entry
        ab = [pts[b][k] - pts[a][k] for k in range(3)]
        ac = [pts[c][k] - pts[a][k] for k in range(3)]
        ad = [pts[d][k] - pts[a][k] for k in range(3)]
        normal = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
        if sum(normal[k] * ad[k] for k in range(3)) > 0:  # normal points into the tet, so flip the winding
            b, c = c, b
        tris.append((a, b, c))
    nodes = sorted({v for tri in tris for v in tri})
    index = {v: i for i, v in enumerate(nodes)}
    return nodes, [index[v] for tri in tris for v in tri]


def leading_number(text):
    """The number at the start of a log value such as '312 ms (writing ...)', or None."""
    m = re.match(r"\s*([0-9.]+)", text or "")
    return to_float(m.group(1)) if m else None


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def scan_outputs(folder):
    """Top-level output files, the final and undeformed VTK files, and the time-series (if any)."""
    out = {"files": [], "final": None, "undeformed": None, "series": None, "steps_dir": None}
    if not os.path.isdir(folder):
        return out
    for entry in sorted(os.scandir(folder), key=lambda e: e.name.lower()):
        if entry.is_file():
            out["files"].append((entry.name, entry.stat().st_size))
            if entry.name.endswith(".vtk.series"):
                out["series"] = entry.path
            elif entry.name.endswith(".vtk"):
                if "undeformed" in entry.name.lower():
                    out["undeformed"] = entry.name
                elif out["final"] is None:
                    out["final"] = entry.name
        elif entry.is_dir():
            vtks = [e for e in os.scandir(entry.path) if e.is_file() and e.name.endswith(".vtk")]
            out["steps_dir"] = (entry.name, len(vtks), sum(e.stat().st_size for e in vtks))
    return out


def read_series(program_dir, series_path):
    """[(time, displacement vectors), ...] from a ParaView .vtk.series file."""
    with open(series_path) as f:
        entries = json.load(f)["files"]
    return [(e["time"], read_vtk_vectors(os.path.join(program_dir, e["name"]))) for e in entries]


# ---------------------------------------------------------------- numbers
def magnitudes(U):
    return [math.sqrt(x * x + y * y + z * z) for x, y, z in U]


def nodal_difference(A, B):
    """Largest and RMS nodal distance between two displacement fields, or None if they don't line up."""
    if not A or not B or len(A) != len(B):
        return None
    d = [math.dist(a, b) for a, b in zip(A, B)]
    return max(d), math.sqrt(sum(v * v for v in d) / len(d))


def nice_ticks(vmax, count=5):
    if vmax <= 0:
        return [0.0, 1.0]
    raw = vmax / count
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    return [i * step for i in range(int(math.ceil(vmax / step - 1e-9)) + 1)]


def fmt_ms(t):
    return "–" if t is None else f"{t:,.0f}"


def fmt_mm(metres):
    """Metres -> millimetres, 3 significant figures."""
    if metres is None:
        return "–"
    v = metres * 1000
    return "0" if v == 0 else f"{v:.3g}"


def fmt_tick(v):
    return f"{v:g}"


def esc(text):
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------- environment
def in_colab():
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:  # no "google" package at all
        return False


def environment(metas, programs):
    env = []
    for kind in RUN_KINDS:
        if kind in metas:
            m = metas[kind]
            took = f", took {m['duration_s']} s including compiling" if m.get("duration_s") else ""
            env.append((f"{kind.upper()} run", f"src/{kind}/run_all_{kind}.sh, finished {m.get('finished', '?')}{took}"))
    for kind in RUN_KINDS:
        m = metas.get(kind, {})
        if m.get("system"):
            parts = [m.get("cpu_model") or "unknown CPU", f"{m['cpu_cores']} cores" if m.get("cpu_cores") else "", m["system"],
                     "Google Colab" if m.get("colab") == "yes" else "", f"GPU: {m['gpu']}" if m.get("gpu") else ""]
            env.append((f"{kind.upper()} run machine", ", ".join(x for x in parts if x)))
    if any(metas.get(k, {}).get("system") for k in RUN_KINDS):
        machine_known = True
    else:
        machine_known = False
    mac = platform.mac_ver()[0]
    system = f"macOS {mac} ({platform.machine()})" if mac else f"{platform.system()} {platform.release()} ({platform.machine()})"
    if os.environ.get("COLAB_RELEASE_TAG") or in_colab():
        system = "Google Colab, " + system
    if not machine_known:
        env.append(("System", system))
    cpu = None
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    if not cpu and os.path.isfile("/proc/cpuinfo"):  # Linux, e.g. Colab
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), None)
    if not machine_known:
        env.append(("CPU", f"{cpu or platform.processor() or platform.machine()}, {os.cpu_count()} cores"))
    if "cpu" in metas:
        env.append(("C++ compiler", f"{metas['cpu'].get('compiler', 'unknown')}, flags: {metas['cpu'].get('flags', '-std=c++17 -O3')}"))
        env.append(("OpenMP threads", metas["cpu"].get("threads", "–")))
    if "gpu" in metas:
        env.append(("CUDA compiler", f"{metas['gpu'].get('nvcc', 'unknown')}, flags: {metas['gpu'].get('flags', '-std=c++17 -O3')}"))
    else:
        env.append(("GPU run", "none yet; run src/gpu/run_all_gpu.sh on a machine with an NVIDIA GPU and the CUDA Toolkit"))
    gpus = sorted({p["info"]["GPU"] for p in programs if p["info"].get("GPU")})
    if gpus:
        env.append(("GPU", ", ".join(gpus)))
    return env


# ---------------------------------------------------------------- HTML pieces
def status_badge(ok, label):
    if not ok and label.startswith("skipped"):  # not run on purpose (e.g. no CUDA on this machine): neutral, not an error
        return f'<span class="status skipped"><span aria-hidden="true">–</span> {esc(label[0].upper() + label[1:])}</span>'
    return (f'<span class="status {"good" if ok else "critical"}"><span aria-hidden="true">{"✓" if ok else "✕"}</span> '
            f'{esc(label)}</span>')


def bar_chart(programs):
    timed = [p for p in programs if p["time_ms"] is not None]
    if not timed:
        return '<p class="muted">No computation times were recorded.</p>'
    vmax = max(p["time_ms"] for p in timed)
    rows = []
    for p in programs:
        t = p["time_ms"]
        series = "s2" if p["anp"] else "s1"
        if t is None:
            rows.append(f'<div class="bar-row"><div class="bar-label">{esc(p["name"])}</div>'
                        f'<div class="bar-track"><span class="bar-val muted">{esc(p["status"])}</span></div></div>')
            continue
        tip = f'{p["name"]}\n{fmt_ms(t)} ms\n{p["formulation"]} formulation, {p["parallel_label"]}'
        if p["timer_note"]:
            tip += f'\n{p["timer_note"]}'
        mark = ' <span class="muted">†</span>' if p["timer_note"] else ""
        rows.append(
            f'<div class="bar-row" tabindex="0" data-tip="{esc(tip)}">'
            f'<div class="bar-label">{esc(p["name"])}</div>'
            f'<div class="bar-track"><div class="bar" style="width:calc((100% - 92px) * {t / vmax:.4f});background:var(--{series})"></div>'
            f'<span class="bar-val">{fmt_ms(t)} ms{mark}</span></div></div>')
    return ('<div class="legend"><span class="key"><i style="background:var(--s1)"></i>Standard formulation</span>'
            '<span class="key"><i style="background:var(--s2)"></i>ANP formulation</span></div>'
            '<div class="bars" role="list">' + "".join(rows) + "</div>")


def speed_section(by_name, metas):
    """CPU vs GPU speed: solver time of each algorithm on serial CPU, OpenMP CPU and GPU, with GPU speed-ups."""
    series = [("Serial CPU", "s1"), ("OpenMP CPU", "s2"), ("GPU", "s3")]
    def solver(name):
        p = by_name.get(name) if name else None
        return p["solver_ms"] if p and p["ok"] and p["solver_ms"] else None
    groups = [(alg, [solver(n) for n in names], names) for alg, *names in SPEED_GROUPS]
    out = ['<section class="card" id="speed"><h2>CPU vs GPU speed</h2>']
    if not any(v[2] for _, v, _ in groups):
        out.append('<p class="secondary">No GPU timings yet. Run <code>src/gpu/run_all_gpu.sh</code> on a machine with an NVIDIA GPU, or the Colab '
                   'notebook (then unzip its download into <code>results/</code> and run <code>python3 scripts/make_report.py</code>), '
                   'and this section compares each algorithm on the CPU and the GPU.</p></section>')
        return "".join(out)
    cpu_m, gpu_m = metas.get("cpu", {}), metas.get("gpu", {})
    cpu_desc = cpu_m.get("cpu_model") or cpu_m.get("system") or "the CPU"
    gpu_desc = gpu_m.get("gpu") or next((p["info"]["GPU"] for p in by_name.values() if p["info"].get("GPU")), "the GPU")
    out.append('<p class="secondary">Solver time: time in the simulation loop, without the time the DJTLED programs spend writing their '
               f'timestep files. CPU: {esc(cpu_desc)}, OpenMP with {esc(cpu_m.get("threads", "?"))} threads. GPU: {esc(gpu_desc)}'
               f'{" on Google Colab" if gpu_m.get("colab") == "yes" else ""}. Lower is faster.</p>')
    vmax = max(v for _, vals, _ in groups for v in vals if v)
    out.append('<div class="legend">' + "".join(f'<span class="key"><i style="background:var(--{c})"></i>{l}</span>' for l, c in series) + "</div>")
    out.append('<div class="bars">')
    for alg, vals, names in groups:
        if not any(vals):
            continue
        out.append(f'<div class="bar-group">{esc(alg)}</div>')
        for (label, color), v, name in zip(series, vals, names):
            if v is None:
                continue
            tip = f"{name}\n{label}: {fmt_ms(v)} ms solver time"
            out.append(f'<div class="bar-row" tabindex="0" data-tip="{esc(tip)}"><div class="bar-label">{label}</div>'
                       f'<div class="bar-track"><div class="bar" style="width:calc((100% - 92px) * {v / vmax:.4f});background:var(--{color})"></div>'
                       f'<span class="bar-val">{fmt_ms(v)} ms</span></div></div>')
    out.append("</div>")
    rows = []
    for alg, (ser, omp, gpu), names in groups:
        if not any((ser, omp, gpu)):
            continue
        sp = lambda cpu: f"{cpu / gpu:.2f}×" if cpu and gpu else "–"
        steps = (by_name.get(names[2]) or {}).get("num_steps")
        per_step = f"{gpu / steps * 1000:.1f}" if gpu and steps else "–"
        rows.append([f"<b>{esc(alg)}</b>", fmt_ms(ser), fmt_ms(omp), fmt_ms(gpu), sp(ser), sp(omp), per_step])
    out.append(table(["Algorithm", "Serial CPU (ms)", "OpenMP CPU (ms)", "GPU (ms)", "GPU speed-up vs serial", "GPU speed-up vs OpenMP",
                      "GPU time per step (µs)"], rows, numeric_cols=(1, 2, 3, 4, 5, 6)))
    out.append('<p class="note">Speed-up above 1× means the GPU is faster. This model is small (1,695 nodes, 7,726 elements), so each time step '
               'gives the GPU little work and the fixed cost of launching its kernels takes a large share of the time; larger meshes '
               'generally favour the GPU more.')
    if cpu_m.get("system") and gpu_m.get("system") and (cpu_m.get("system"), cpu_m.get("cpu_model")) != (gpu_m.get("system"), gpu_m.get("cpu_model")):
        out.append(' The CPU and GPU runs were on different machines (see Environment).')
    out.append("</p></section>")
    return "".join(out)


def line_chart(points, label):
    """points: [(time_s, value_m)]. Inline SVG plotted in mm, with a crosshair tooltip (see the script at the end of the page)."""
    W, H, L, R, T, B = 720, 260, 56, 64, 14, 40
    tmax = max(t for t, _ in points) or 1.0
    ticks = nice_ticks(max(v for _, v in points) * 1000)
    ymax = ticks[-1] or 1.0
    x = lambda t: L + (W - L - R) * t / tmax
    y = lambda mm: T + (H - T - B) * (1 - mm / ymax)
    parts = [f'<svg class="line-chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(label)} over time, in millimetres" data-label="{esc(label)}" data-points=\'']
    data = [[round(x(t), 2), round(y(v * 1000), 2), round(t, 4), fmt_mm(v)] for t, v in points]
    parts[0] += esc(json.dumps(data)) + "'>"
    for tk in ticks:
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(tk):.1f}" y2="{y(tk):.1f}"/>'
                     f'<text class="tick" x="{L - 8}" y="{y(tk) + 4:.1f}" text-anchor="end">{fmt_tick(tk)}</text>')
    parts.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    for tk in nice_ticks(tmax, 5):
        if tk <= tmax * 1.01:
            parts.append(f'<text class="tick" x="{x(tk):.1f}" y="{H - B + 18}" text-anchor="middle">{fmt_tick(tk)}</text>')
    parts.append(f'<text class="axis-title" x="{(L + W - R) / 2}" y="{H - 4}" text-anchor="middle">Time (s)</text>')
    parts.append(f'<text class="axis-title" x="12" y="{(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 12 {(T + H - B) / 2})">mm</text>')
    path = " ".join(f'{"M" if i == 0 else "L"}{x(t):.1f},{y(v * 1000):.1f}' for i, (t, v) in enumerate(points))
    parts.append(f'<path class="series-line" d="{path}"/>')
    lx, ly = x(points[-1][0]), y(points[-1][1] * 1000)
    parts.append(f'<circle class="end-dot" cx="{lx:.1f}" cy="{ly:.1f}" r="4"/>'
                 f'<text class="end-label" x="{lx + 8:.1f}" y="{ly + 4:.1f}">{fmt_mm(points[-1][1])} mm</text>')
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{T}" y2="{H - B}" visibility="hidden"/>'
                 f'<circle class="hover-dot" r="4" cx="0" cy="0" visibility="hidden"/>'
                 f'<rect class="hit" x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}"/></svg>')
    return "".join(parts)


def table(headers, rows, numeric_cols=(), cls=""):
    head = "".join(f'<th{" class=num" if i in numeric_cols else ""}>{h}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td{" class=num" if i in numeric_cols else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# ---------------------------------------------------------------- animation
def animation_payload(programs):
    """Surface mesh and per-timestep displacements of every program that saved a time series.
    Displacements are stored as 16-bit integers scaled to the largest component (about 0.002 mm resolution
    for 50 mm), base64-encoded, so the page stays a single file of a few MB."""
    candidates = [p for p in programs if p["series_data"] and p["outputs"]["undeformed"]]
    if not candidates:
        return None
    times = [t for t, _ in candidates[0]["series_data"]]
    entries, keys = [], []
    for p in candidates:
        sd = p["series_data"]
        if len(sd) != len(times) or any(abs(t - t0) > 1e-9 for (t, _), t0 in zip(sd, times)) or any(U is None for _, U in sd):
            continue  # all views share one time slider, so they need the same timesteps
        pts, tets = read_vtk_mesh(os.path.join(p["folder"], p["outputs"]["undeformed"]))
        if not pts or not tets or any(len(U) != len(pts) for _, U in sd):
            continue
        nodes, tris = surface_mesh(pts, tets)
        scale = max((abs(c) for _, U in sd for i in nodes for c in U[i]), default=0.0) or 1.0
        q = array.array("h", (int(round(c / scale * 32767)) for _, U in sd for i in nodes for c in U[i]))
        if sys.byteorder == "big":
            q.byteswap()  # the page reads little-endian
        entries.append({"name": p["name"], "formulation": p["formulation"],
                        "pos": [round(c, 7) for i in nodes for c in pts[i]], "tris": tris,
                        "scale": scale, "frames": base64.b64encode(q.tobytes()).decode("ascii")})
        keys.append((tuple(nodes), tuple(tris)))
    if not entries:
        return None
    index = {e["name"]: i for i, e in enumerate(entries)}
    diffs = [{"a": index[a], "b": index[b], "label": f"Difference, {a} vs {b}"}
             for a, b, _, _ in PAIRS if a in index and b in index and keys[index[a]] == keys[index[b]]]
    return {"times": times, "programs": entries, "diffs": diffs}


def animation_section(payload, programs):
    names = ", ".join(e["name"] for e in payload["programs"])
    others = [p["name"] for p in programs if p["ok"] and p["name"] not in {e["name"] for e in payload["programs"]}]
    note = f" The other programs ({', '.join(others)}) save only the final state, so they have no animation." if others else ""
    options = '<option value="u">Displacement |u|</option>' + "".join(
        f'<option value="{i}">{esc(d["label"])}</option>' for i, d in enumerate(payload["diffs"]))
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return ('<section class="card" id="animation"><h2>Deformation animation</h2>'
            f'<p class="secondary">Outer surface of the mesh at each saved timestep of {esc(names)}.{esc(note)} '
            'Drag to rotate, pinch or hold ⌘/Ctrl and scroll to zoom; all views move together. '
            'With a view selected, the arrow keys rotate it and Space plays or pauses.</p>'
            '<div class="anim-controls">'
            '<button type="button" id="anim-play">Pause</button>'
            f'<input type="range" id="anim-slider" min="0" max="{len(payload["times"]) - 1}" value="0" aria-label="Timestep">'
            '<span id="anim-time" class="anim-time">t = 0 s</span>'
            f'<label class="anim-mode">Colour <select id="anim-mode">{options}</select></label>'
            '<button type="button" id="anim-reset">Reset view</button></div>'
            '<div class="anim-grid" id="anim-grid"></div>'
            '<div class="colorbar"><div class="cb-title" id="cb-title"></div><div class="cb-bar" id="cb-bar"></div>'
            '<div class="cb-ticks" id="cb-ticks"></div></div>'
            f'<script type="application/json" id="anim-data">{data}</script></section>')


# ---------------------------------------------------------------- report
def build(results):
    metas, rows = read_summaries(results)
    meta = metas.get("cpu", {})  # OpenMP thread count
    programs = []
    for r in rows:
        folder = os.path.join(results, r["name"])
        info, log_lines = parse_log(os.path.join(folder, "log.txt"))
        outputs = scan_outputs(folder)
        # a failed program may have left files from an earlier run, so only read results of programs that succeeded
        U = read_vtk_vectors(os.path.join(folder, outputs["final"])) if outputs["final"] and r["status"] == "ok" else None
        mags = magnitudes(U) if U else None
        series_data = None
        if outputs["series"] and r["status"] == "ok":
            try:
                series_data = read_series(folder, outputs["series"])
            except (OSError, ValueError, KeyError):
                series_data = None
        # Logs from before the timers were separated (their "Export time" says it is "included in the computation time")
        # count the timestep VTK files in the computation time; newer logs time the export separately.
        export_included = "; included in the computation time" in (info.get("Export time") or "")  # newer logs say "; not included ..."
        timer_note = None
        if outputs["steps_dir"] and export_included:
            timer_note = f"Time includes writing {outputs['steps_dir'][1]} timestep VTK files"
        programs.append(dict(
            r, folder=folder, info=info, log=log_lines, outputs=outputs, U=U, series_data=series_data,
            ok=r["status"] == "ok",
            anp=r["name"].startswith("ANP"),
            formulation="ANP" if r["name"].startswith("ANP") else "Standard",
            algorithm="Direct Jacobian TLED" if "DJTLED" in r["name"] else "TLED",
            parallel_label=("CUDA GPU" if r["name"].endswith("_CUDA") else
                            f"OpenMP, {info.get('Threads', meta.get('threads', '?'))} threads" if r["openmp"] else "Serial"),
            skipped=r["status"].startswith("skipped"),
            max_u=max(mags) if mags else None,
            mean_u=sum(mags) / len(mags) if mags else None,
            timer_note=timer_note,
            export_ms=leading_number(info.get("Export time")),
            export_included=export_included,
            num_steps=leading_number(info.get("NumSteps"))))
    for p in programs:
        p["solver_ms"] = (p["time_ms"] - p["export_ms"]) if p["export_included"] and p["time_ms"] is not None and p["export_ms"] is not None else p["time_ms"]
    by_name = {p["name"]: p for p in programs}

    # ---- headline tiles
    n_ok = sum(p["ok"] for p in programs)
    skipped = [p["name"] for p in programs if p["skipped"]]
    n_ran = len(programs) - len(skipped)
    sub = "All ran without errors" if n_ok == n_ran else "See the status column below"
    if skipped:
        sub += f"; skipped: {', '.join(skipped)}"
    tiles = [("Programs completed", f"{n_ok} of {n_ran}", sub)]
    timed = [p for p in programs if p["ok"] and p["time_ms"] is not None]
    if timed:
        fastest = min(timed, key=lambda p: p["time_ms"])
        tiles.append(("Fastest program", f"{fastest['name']}", f"{fmt_ms(fastest['time_ms'])} ms"))
    def ratio(a, b):
        pa, pb = by_name.get(a), by_name.get(b)
        if pa and pb and pa["ok"] and pb["ok"] and pa["time_ms"] and pb["time_ms"]:
            return pa["time_ms"] / pb["time_ms"]
        return None
    r = ratio("TLED", "TLED_OMP")
    if r:
        tiles.append(("OpenMP speed-up, TLED", f"{r:.2f}×", f"TLED_OMP vs TLED, {by_name['TLED_OMP']['info'].get('Threads', meta.get('threads', '?'))} threads"))
    r = ratio("ANPTLED", "TLED")
    if r:
        tiles.append(("Time cost of ANP, TLED", f"{r:.2f}×", "ANPTLED vs TLED, both serial"))
    r = ratio("TLED", "TLED_CUDA")
    if r:
        tiles.append(("GPU speed-up, TLED", f"{r:.2f}×", "TLED_CUDA vs serial TLED"))

    # ---- results table
    base = by_name.get(BASELINE)
    base_t = base["time_ms"] if base and base["ok"] else None
    result_rows = []
    for p in programs:
        speed = f"{base_t / p['time_ms']:.2f}×" if base_t and p["time_ms"] else "–"
        time_cell = fmt_ms(p["time_ms"]) + (' <span class="muted">†</span>' if p["timer_note"] else "")
        result_rows.append([f'<b>{esc(p["name"])}</b>', esc(p["algorithm"]), esc(p["formulation"]), esc(p["parallel_label"]),
                            esc(os.path.basename(p["model"])), status_badge(p["ok"], "OK" if p["ok"] else p["status"]),
                            time_cell, fmt_ms(p["export_ms"]), speed, fmt_mm(p["max_u"]), fmt_mm(p["mean_u"])])

    # ---- comparisons
    comp_rows, histories = [], []
    for a, b, expect_match, purpose in PAIRS:
        pa, pb = by_name.get(a), by_name.get(b)
        if not pa or not pb:
            continue
        d = nodal_difference(pa["U"], pb["U"])
        if d is None:
            not_run = [p["name"] for p in (pa, pb) if p["skipped"]]
            why = f"{', '.join(not_run)} was skipped" if not_run else "No results to compare"
            comp_rows.append([f"{esc(a)} vs {esc(b)}", esc(purpose), "–", "–", "–", f'<span class="muted">{esc(why)}</span>'])
            continue
        scale = max(pa["max_u"] or 0, pb["max_u"] or 0) or 1.0
        rel = d[0] / scale
        if expect_match:
            verdict = status_badge(rel <= MATCH_TOL, "Match" if rel <= MATCH_TOL else "Differs")
        else:
            verdict = '<span class="secondary">Formulation effect</span>'
        comp_rows.append([f"<b>{esc(a)}</b> vs <b>{esc(b)}</b>", esc(purpose), fmt_mm(d[0]), fmt_mm(d[1]), f"{rel * 100:.3g}%", verdict])
        A, Bs = pa["series_data"], pb["series_data"]
        if A and Bs and not expect_match:
            if len(A) == len(Bs) and all(abs(ta - tb) < 1e-9 for (ta, _), (tb, _) in zip(A, Bs)):
                pts = [(ta, nodal_difference(ua, ub)[0]) for (ta, ua), (_, ub) in zip(A, Bs) if nodal_difference(ua, ub)]
                if pts:
                    histories.append((a, b, purpose, pts))

    # ---- model settings, one column per model file
    models = {}
    for p in programs:
        models.setdefault(os.path.basename(p["model"]), p["info"])
    setting_labels = [("Nodes", "Nodes"), ("Elements", "Elements"), ("EleMaterial", "Material (μ, κ in Pa) and density (kg/m³)"),
                      ("BC", "Boundary conditions"), ("DampingCoef.", "Damping coefficient"), ("TimeStep", "Time step (s)"),
                      ("TotalTime", "Total time (s)"), ("NumSteps", "Steps")]
    model_rows = [[esc(label)] + [esc(info.get(key, "–")) for info in models.values()] for key, label in setting_labels]

    # ---- assemble
    env = environment(metas, programs)
    out = [PAGE_HEAD]
    out.append('<header><h1>ANP-DJTLED run report</h1>'
               '<p class="secondary">Results from ' + " and ".join(
                   f'<code>src/{k}/run_all_{k}.sh</code> (finished {esc(metas[k].get("finished", "?"))})' for k in RUN_KINDS if k in metas) +
               '.</p></header>')
    out.append('<section class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{esc(l)}</div><div class="tile-value">{esc(v)}</div><div class="tile-sub">{esc(s)}</div></div>'
        for l, v, s in tiles) + "</section>")

    out.append('<section class="card"><h2>Computation time</h2>'
               '<p class="secondary">Time spent in each program\'s simulation loop, as printed in its log. Lower is faster.</p>')
    out.append(bar_chart(programs))
    if any(p["timer_note"] for p in programs):
        out.append('<p class="note">† The DJTLED timers also include writing the timestep VTK files during the loop. '
                   'The TLED timers stop before any file is written, so the two families are not a like-for-like comparison.</p>')
    out.append("</section>")

    out.append(speed_section(by_name, metas))

    out.append('<section class="card"><h2>Results by program</h2>'
               f'<p class="secondary">Time is the time-step loop only; VTK export is the time spent writing the animation frames and the final VTK files, timed separately. '
               f'Speed-up is relative to <b>{esc(BASELINE)}</b>; above 1× is faster. '
               'Displacements are the final nodal displacement magnitudes, |u|.</p>')
    out.append(table(["Program", "Algorithm", "Formulation", "Parallel", "Model", "Status", "Time (ms)", "VTK export (ms)", f"Speed-up vs {esc(BASELINE)}", "Max |u| (mm)", "Mean |u| (mm)"],
                     result_rows, numeric_cols=(6, 7, 8, 9, 10)))
    out.append("</section>")

    if comp_rows:
        out.append('<section class="card"><h2>Agreement between programs</h2>'
                   '<p class="secondary">Distance between the two programs\' final displacements at each node: the largest, and the root mean square over all nodes. '
                   f'Programs with the same formulation should match, within {MATCH_TOL * 100:g}% of the largest displacement. '
                   'The ANP rows show how much the ANP formulation changes the result.</p>')
        out.append(table(["Comparison", "What it shows", "Max difference (mm)", "RMS difference (mm)", "Max as % of max |u|", "Result"],
                         comp_rows, numeric_cols=(2, 3, 4)))
        out.append('<p class="note">VTK files store values to 6 significant figures, so differences below about 0.001 mm are rounding.</p></section>')

    for a, b, purpose, pts in histories:
        label = f"Max difference, {a} vs {b}"
        out.append(f'<section class="card"><h2>{esc(purpose)}, over time</h2>'
                   f'<p class="secondary">Largest nodal distance between <b>{esc(a)}</b> and <b>{esc(b)}</b> at each exported timestep, in mm.</p>')
        out.append('<div class="chart-wrap">' + line_chart(pts, label) + "</div>")
        out.append('<details><summary>Show as table</summary>' +
                   table(["Time (s)", "Max difference (mm)"], [[f"{t:g}", fmt_mm(v)] for t, v in pts], numeric_cols=(0, 1)) +
                   "</details></section>")

    payload = animation_payload(programs)
    if payload:
        out.append(animation_section(payload, programs))

    out.append('<section class="card"><h2>Model and simulation settings</h2>'
               '<p class="secondary">As printed by the programs when they read each model file.</p>')
    out.append(table(["Setting"] + [esc(m) for m in models], model_rows))
    out.append("</section>")

    out.append('<section class="card"><h2>Environment</h2>')
    out.append('<dl class="env">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in env) + "</dl></section>")

    out.append('<section class="card"><h2>Program details</h2><p class="secondary">Output files and the end of each program\'s log.</p>')
    for p in programs:
        files = [[f"<code>{esc(n)}</code>", human_size(s)] for n, s in p["outputs"]["files"]]
        if p["outputs"]["steps_dir"]:
            d, n, s = p["outputs"]["steps_dir"]
            files.append([f"<code>{esc(d)}/</code> ({n} timestep files)", human_size(s)])
        out.append(f'<details><summary><b>{esc(p["name"])}</b> {status_badge(p["ok"], "OK" if p["ok"] else p["status"])}</summary>'
                   f'<p class="secondary">Folder: <code>{esc(os.path.relpath(p["folder"]))}</code></p>'
                   + table(["File", "Size"], files, numeric_cols=(1,))
                   + f'<pre>{esc(chr(10).join(p["log"][-15:]))}</pre></details>')
    out.append("</section>")

    out.append(f'<footer class="muted">Generated by <code>scripts/make_report.py</code> on {datetime.datetime.now():%Y-%m-%d %H:%M:%S}.</footer>')
    out.append(PAGE_TAIL)
    return "".join(out)


PAGE_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ANP-DJTLED run report</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --text: #0b0b0b; --text-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --good: #006300; --critical: #d03b3b; --code-bg: #f0efec;
  /* sequential blue ramp for the animation, low -> high (light mode: light to dark) */
  --ramp: #b7d3f6,#9ec5f4,#86b6ef,#6da7ec,#5598e7,#3987e5,#2a78d6,#256abf,#1c5cab,#184f95,#104281,#0d366b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --text: #f0efec; --text-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --good: #0ca30c; --critical: #e66767; --code-bg: #262624;
    --ramp: #184f95,#1c5cab,#256abf,#2a78d6,#3987e5,#5598e7,#6da7ec,#86b6ef,#9ec5f4,#b7d3f6,#cde2fb;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --text: #f0efec; --text-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --good: #0ca30c; --critical: #e66767; --code-bg: #262624;
  --ramp: #184f95,#1c5cab,#256abf,#2a78d6,#3987e5,#5598e7,#6da7ec,#86b6ef,#9ec5f4,#b7d3f6,#cde2fb;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 16px; margin: 0 0 4px; font-weight: 600; }
p { margin: 0 0 12px; }
code { font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--code-bg); padding: 1px 4px; border-radius: 4px; }
.secondary { color: var(--text-2); }
.muted { color: var(--muted); }
.note { color: var(--text-2); font-size: 13px; margin: 12px 0 0; }
header { margin-bottom: 20px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.tile-label { color: var(--text-2); font-size: 13px; }
.tile-value { font-size: 26px; font-weight: 600; margin: 2px 0; overflow-wrap: anywhere; }
.tile-sub { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 12px; color: var(--text-2); font-size: 13px; }
.key { display: inline-flex; align-items: center; gap: 6px; }
.key i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.bars { display: flex; flex-direction: column; gap: 6px; }
.bar-group { font-size: 13px; font-weight: 600; margin: 10px 0 2px; }
.bar-group:first-child { margin-top: 0; }
.bar-row { display: flex; align-items: center; gap: 12px; border-radius: 6px; outline-offset: 2px; }
.bar-row:hover .bar, .bar-row:focus .bar { filter: brightness(1.08); }
.bar-label { width: 120px; flex: none; font-size: 13px; text-align: right; color: var(--text); font-variant-numeric: tabular-nums; }
.bar-track { flex: 1; display: flex; align-items: center; gap: 8px; min-width: 0; height: 28px; border-left: 1px solid var(--axis); }
.bar { height: 20px; border-radius: 0 4px 4px 0; min-width: 2px; }
.bar-val { font-size: 13px; color: var(--text-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; font-weight: 600; color: var(--text-2); border-bottom: 1px solid var(--axis); padding: 6px 10px; white-space: nowrap; }
td { border-bottom: 1px solid var(--grid); padding: 6px 10px; vertical-align: top; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.status { display: inline-flex; gap: 4px; align-items: center; font-weight: 600; white-space: nowrap; }
.status.good { color: var(--good); }
.status.critical { color: var(--critical); }
.status.skipped { color: var(--text-2); font-weight: 400; }
.chart-wrap { overflow-x: auto; }
.line-chart { width: 100%; min-width: 480px; height: auto; display: block; }
.line-chart .grid { stroke: var(--grid); stroke-width: 1; }
.line-chart .axis { stroke: var(--axis); stroke-width: 1; }
.line-chart .tick, .line-chart .axis-title { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.line-chart .series-line { fill: none; stroke: var(--s1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.line-chart .end-dot, .line-chart .hover-dot { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
.line-chart .end-label { fill: var(--text-2); font-size: 12px; }
.line-chart .cross { stroke: var(--axis); stroke-width: 1; }
.line-chart .hit { fill: transparent; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--text-2); padding: 4px 0; }
details > .table-wrap, details > pre, details > p { margin-top: 8px; }
pre { background: var(--code-bg); border-radius: 8px; padding: 12px; overflow-x: auto; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
.env { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 8px 0 0; }
.env dt { color: var(--text-2); }
.env dd { margin: 0; overflow-wrap: anywhere; }
#tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--text); border: 1px solid var(--border);
       box-shadow: 0 4px 16px rgba(0,0,0,0.15); border-radius: 8px; padding: 6px 10px; font-size: 12px; white-space: pre-line; max-width: 280px; }
footer { font-size: 12px; margin-top: 24px; }
button, select { font: inherit; color: var(--text); background: var(--surface); border: 1px solid var(--axis); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
button:hover, select:hover { border-color: var(--muted); }
.anim-controls { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin: 8px 0 12px; }
.anim-controls input[type=range] { flex: 1 1 200px; accent-color: var(--s1); }
.anim-time { min-width: 150px; color: var(--text-2); font-variant-numeric: tabular-nums; }
.anim-mode { display: inline-flex; gap: 6px; align-items: center; color: var(--text-2); }
.anim-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.anim-view { margin: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.anim-view figcaption { padding: 6px 10px; font-size: 13px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.anim-view figcaption .anim-max { color: var(--text-2); font-variant-numeric: tabular-nums; }
.anim-view canvas { display: block; width: 100%; aspect-ratio: 4 / 3; touch-action: none; cursor: grab; outline-offset: -2px; }
.anim-view canvas:active { cursor: grabbing; }
.anim-view .anim-error { padding: 24px 12px; color: var(--text-2); }
.colorbar { margin-top: 14px; max-width: 420px; }
.cb-title { font-size: 13px; color: var(--text-2); margin-bottom: 4px; }
.cb-bar { height: 10px; border-radius: 2px; }
.cb-ticks { position: relative; height: 18px; font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
.cb-ticks span { position: absolute; top: 2px; transform: translateX(-50%); }
.cb-ticks span:first-child { transform: none; }
.cb-ticks span:last-child { transform: translateX(-100%); }
@media (max-width: 560px) { .bar-label { width: 96px; } .env { grid-template-columns: 1fr; } .env dd { margin-bottom: 6px; } }
</style></head><body><main>
"""

PAGE_TAIL = """</main><div id="tip" hidden></div>
<script>
(function () {
  const tip = document.getElementById("tip");
  function show(text, x, y) {
    tip.textContent = text; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(8, Math.min(x + 14, innerWidth - w - 8)) + "px";
    tip.style.top = Math.max(8, Math.min(y + 14, innerHeight - h - 8)) + "px";
  }
  function hide() { tip.hidden = true; }
  document.querySelectorAll("[data-tip]").forEach(function (el) {
    el.addEventListener("mousemove", function (e) { show(el.dataset.tip, e.clientX, e.clientY); });
    el.addEventListener("mouseleave", hide);
    el.addEventListener("focus", function () { const r = el.getBoundingClientRect(); show(el.dataset.tip, r.left + 120, r.bottom); });
    el.addEventListener("blur", hide);
  });
  document.querySelectorAll("svg.line-chart").forEach(function (svg) {
    const pts = JSON.parse(svg.dataset.points), cross = svg.querySelector(".cross"), dot = svg.querySelector(".hover-dot");
    svg.addEventListener("mousemove", function (e) {
      const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
      const q = p.matrixTransform(svg.getScreenCTM().inverse());
      let best = pts[0];
      for (const pt of pts) if (Math.abs(pt[0] - q.x) < Math.abs(best[0] - q.x)) best = pt;
      cross.setAttribute("x1", best[0]); cross.setAttribute("x2", best[0]); cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", best[0]); dot.setAttribute("cy", best[1]); dot.setAttribute("visibility", "visible");
      show("t = " + best[2] + " s\\n" + svg.dataset.label + ": " + best[3] + " mm", e.clientX, e.clientY);
    });
    svg.addEventListener("mouseleave", function () { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hide(); });
  });
})();

// ---- deformation animation: a small WebGL viewer, no external libraries
(function () {
  const dataEl = document.getElementById("anim-data");
  if (!dataEl) return;
  const D = JSON.parse(dataEl.textContent), nF = D.times.length;
  const grid = document.getElementById("anim-grid"), slider = document.getElementById("anim-slider"),
        playBtn = document.getElementById("anim-play"), timeEl = document.getElementById("anim-time"),
        modeSel = document.getElementById("anim-mode"), resetBtn = document.getElementById("anim-reset");
  const HOME = { yaw: 0.6, pitch: 0.35, zoom: 1 };
  const state = { frame: 0, playing: false, yaw: HOME.yaw, pitch: HOME.pitch, zoom: HOME.zoom, mode: "u" };
  const clamp = function (v, a, b) { return Math.min(b, Math.max(a, v)); };

  // displacements: little-endian int16, scaled so 32767 = the largest displacement component
  const progs = D.programs.map(function (p) {
    const bin = atob(p.frames), u = new Float32Array(bin.length / 2), f = p.scale / 32767;
    for (let i = 0; i < u.length; i++) {
      let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      if (v > 32767) v -= 65536;
      u[i] = v * f;
    }
    return { name: p.name, formulation: p.formulation, pos: p.pos, tris: p.tris, n: p.pos.length / 3, u: u, scale: p.scale };
  });

  // value shown as colour at each surface node: |u| of the view's own program, or |uA - uB| for a difference
  function scalars(P, f) {
    const s = new Float32Array(P.n);
    if (state.mode === "u") {
      const o = f * P.n * 3, u = P.u;
      for (let i = 0; i < P.n; i++) s[i] = Math.hypot(u[o + 3 * i], u[o + 3 * i + 1], u[o + 3 * i + 2]);
    } else {
      const d = D.diffs[+state.mode], A = progs[d.a].u, B = progs[d.b].u, o = f * P.n * 3;
      for (let i = 0; i < P.n; i++) {
        const j = o + 3 * i;
        s[i] = Math.hypot(A[j] - B[j], A[j + 1] - B[j + 1], A[j + 2] - B[j + 2]);
      }
    }
    return s;
  }
  const maxCache = {};
  function modeMax() { // over all timesteps and views, so the colour scale stays fixed while playing
    if (maxCache[state.mode] === undefined) {
      let m = 0;
      (state.mode === "u" ? progs : [progs[D.diffs[+state.mode].a]]).forEach(function (P) {
        for (let f = 0; f < nF; f++) { const s = scalars(P, f); for (let i = 0; i < s.length; i++) if (s[i] > m) m = s[i]; }
      });
      maxCache[state.mode] = m;
    }
    return maxCache[state.mode];
  }
  function niceTicks(vmax) {
    if (!(vmax > 0)) return [0, 1];
    const raw = vmax / 4, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(function (s) { return s * mag; }).find(function (s) { return s >= raw; });
    const out = [];
    for (let v = 0; v < vmax + step * 0.999; v += step) out.push(+v.toPrecision(6));
    return out;
  }
  function fmtMM(m) { const v = m * 1000; return v === 0 ? "0" : String(+v.toPrecision(3)); }

  // colours come from CSS tokens, so the viewer follows light/dark mode
  let ramp = [], surface = [1, 1, 1], scaleTop = 1;
  function hex(h) { h = h.trim().replace("#", ""); return [0, 2, 4].map(function (i) { return parseInt(h.slice(i, i + 2), 16) / 255; }); }
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    ramp = cs.getPropertyValue("--ramp").split(",").map(hex);
    surface = hex(cs.getPropertyValue("--surface"));
  }
  function colour(t, out, k) {
    t = clamp(t, 0, 1) * (ramp.length - 1);
    const i = Math.min(ramp.length - 2, Math.floor(t)), w = t - i;
    for (let c = 0; c < 3; c++) out[k + c] = ramp[i][c] * (1 - w) + ramp[i + 1][c] * w;
  }

  // 4x4 matrices, column-major as WebGL expects
  function mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; }
    return o;
  }
  function persp(fovy, asp, near, far) { const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far); return new Float32Array([f / asp, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]); }
  function trans(x, y, z) { return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]); }
  function rotX(a) { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); }
  function rotY(a) { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); }

  const VS = "attribute vec3 aPos; attribute vec3 aNrm; attribute vec3 aCol; uniform mat4 uMVP; uniform mat3 uNM;" +
             "varying vec3 vCol; varying float vLit;" +
             "void main() { gl_Position = uMVP * vec4(aPos, 1.0); vec3 n = normalize(uNM * aNrm);" +
             "  vLit = 0.7 + 0.3 * abs(dot(n, normalize(vec3(0.35, 0.5, 1.0)))); vCol = aCol; }";
  const FS = "precision mediump float; varying vec3 vCol; varying float vLit; void main() { gl_FragColor = vec4(vCol * vLit, 1.0); }";

  function makeViewer(P) {
    const fig = document.createElement("figure"), cap = document.createElement("figcaption"),
          title = document.createElement("span"), name = document.createElement("b"), maxEl = document.createElement("span");
    fig.className = "anim-view"; maxEl.className = "anim-max";
    name.textContent = P.name;
    title.appendChild(name); title.appendChild(document.createTextNode(" · " + P.formulation + " formulation"));
    cap.appendChild(title); cap.appendChild(maxEl); fig.appendChild(cap);
    const canvas = document.createElement("canvas");
    canvas.tabIndex = 0; canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "3D view of the deforming mesh from " + P.name);
    fig.appendChild(canvas); grid.appendChild(fig);
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) {
      canvas.remove();
      const e = document.createElement("div"); e.className = "anim-error";
      e.textContent = "This browser cannot show 3D graphics (WebGL is turned off or unavailable).";
      fig.appendChild(e); return null;
    }
    const prog = gl.createProgram();
    [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, FS]].forEach(function (s) {
      const sh = gl.createShader(s[0]); gl.shaderSource(sh, s[1]); gl.compileShader(sh); gl.attachShader(prog, sh);
    });
    gl.linkProgram(prog);
    const nV = P.tris.length, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < P.n; i++) for (let c = 0; c < 3; c++) { lo[c] = Math.min(lo[c], P.pos[3 * i + c]); hi[c] = Math.max(hi[c], P.pos[3 * i + c]); }
    const V = {
      P: P, canvas: canvas, gl: gl, prog: prog, maxEl: maxEl, nV: nV,
      center: [0, 1, 2].map(function (c) { return (lo[c] + hi[c]) / 2; }),
      radius: (Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2 || 1) + P.scale, // room for the deformation
      loc: { pos: gl.getAttribLocation(prog, "aPos"), nrm: gl.getAttribLocation(prog, "aNrm"), col: gl.getAttribLocation(prog, "aCol"),
             mvp: gl.getUniformLocation(prog, "uMVP"), nm: gl.getUniformLocation(prog, "uNM") },
      buf: { pos: gl.createBuffer(), nrm: gl.createBuffer(), col: gl.createBuffer() },
      arr: { pos: new Float32Array(nV * 3), nrm: new Float32Array(nV * 3), col: new Float32Array(nV * 3) }
    };
    attachControls(canvas);
    return V;
  }

  function update(V) { // deformed positions, flat normals and colours for the current timestep
    const P = V.P, o = state.frame * P.n * 3, s = scalars(P, state.frame), A = V.arr, dp = new Float32Array(P.n * 3);
    for (let i = 0; i < P.n * 3; i++) dp[i] = P.pos[i] + P.u[o + i];
    let smax = 0;
    for (let i = 0; i < s.length; i++) if (s[i] > smax) smax = s[i];
    let k = 0;
    for (let t = 0; t < P.tris.length; t += 3) {
      const a = P.tris[t] * 3, b = P.tris[t + 1] * 3, c = P.tris[t + 2] * 3;
      const ux = dp[b] - dp[a], uy = dp[b + 1] - dp[a + 1], uz = dp[b + 2] - dp[a + 2];
      const vx = dp[c] - dp[a], vy = dp[c + 1] - dp[a + 1], vz = dp[c + 2] - dp[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (const idx of [a, b, c]) {
        A.pos[k] = dp[idx]; A.pos[k + 1] = dp[idx + 1]; A.pos[k + 2] = dp[idx + 2];
        A.nrm[k] = nx; A.nrm[k + 1] = ny; A.nrm[k + 2] = nz;
        colour(s[idx / 3] / scaleTop, A.col, k);
        k += 3;
      }
    }
    const gl = V.gl;
    ["pos", "nrm", "col"].forEach(function (key) { gl.bindBuffer(gl.ARRAY_BUFFER, V.buf[key]); gl.bufferData(gl.ARRAY_BUFFER, A[key], gl.DYNAMIC_DRAW); });
    V.maxEl.textContent = (state.mode === "u" ? "max |u| " : "max difference ") + fmtMM(smax) + " mm";
  }

  function draw(V) {
    const gl = V.gl, c = V.canvas, dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clearColor(surface[0], surface[1], surface[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    const fov = 0.6, dist = V.radius / Math.sin(fov / 2) / state.zoom;
    const R = mul(rotX(state.pitch), mul(rotY(state.yaw), rotX(-Math.PI / 2))); // -90 deg about x puts the model's z axis up
    const MV = mul(trans(0, 0, -dist), mul(R, trans(-V.center[0], -V.center[1], -V.center[2])));
    const MVP = mul(persp(fov, w / h, Math.max(dist - 2 * V.radius, dist * 0.01), dist + 2 * V.radius), MV);
    gl.useProgram(V.prog);
    gl.uniformMatrix4fv(V.loc.mvp, false, MVP);
    gl.uniformMatrix3fv(V.loc.nm, false, new Float32Array([R[0], R[1], R[2], R[4], R[5], R[6], R[8], R[9], R[10]]));
    ["pos", "nrm", "col"].forEach(function (key) {
      gl.bindBuffer(gl.ARRAY_BUFFER, V.buf[key]); gl.enableVertexAttribArray(V.loc[key]); gl.vertexAttribPointer(V.loc[key], 3, gl.FLOAT, false, 0, 0);
    });
    gl.drawArrays(gl.TRIANGLES, 0, V.nV);
  }

  let views = [];
  function render(geometry) { if (geometry) views.forEach(update); views.forEach(draw); }

  function attachControls(canvas) {
    let drag = null;
    canvas.addEventListener("pointerdown", function (e) { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointermove", function (e) {
      if (!drag) return;
      state.yaw += (e.clientX - drag[0]) * 0.01;
      state.pitch = clamp(state.pitch + (e.clientY - drag[1]) * 0.01, -1.55, 1.55);
      drag = [e.clientX, e.clientY]; render(false);
    });
    canvas.addEventListener("pointerup", function () { drag = null; });
    canvas.addEventListener("pointercancel", function () { drag = null; });
    canvas.addEventListener("wheel", function (e) { // pinch, or Ctrl/Cmd + scroll; plain scrolling still scrolls the page
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault(); state.zoom = clamp(state.zoom * Math.exp(-e.deltaY * 0.01), 0.4, 6); render(false);
    }, { passive: false });
    canvas.addEventListener("keydown", function (e) {
      if (e.key === "ArrowLeft") state.yaw -= 0.1;
      else if (e.key === "ArrowRight") state.yaw += 0.1;
      else if (e.key === "ArrowUp") state.pitch = clamp(state.pitch - 0.1, -1.55, 1.55);
      else if (e.key === "ArrowDown") state.pitch = clamp(state.pitch + 0.1, -1.55, 1.55);
      else if (e.key === " ") setPlaying(!state.playing);
      else return;
      e.preventDefault(); render(false);
    });
  }

  function updateColorbar() {
    const ticks = niceTicks(modeMax() * 1000), top = ticks[ticks.length - 1];
    scaleTop = top / 1000 || 1;
    document.getElementById("cb-title").textContent = (state.mode === "u" ? "Displacement |u|" : D.diffs[+state.mode].label) + " (mm)";
    document.getElementById("cb-bar").style.background = "linear-gradient(to right, " + ramp.map(function (c) {
      return "rgb(" + c.map(function (v) { return Math.round(v * 255); }).join(",") + ")"; }).join(", ") + ")";
    const box = document.getElementById("cb-ticks");
    box.textContent = "";
    ticks.forEach(function (t) { const s = document.createElement("span"); s.style.left = (t / top * 100) + "%"; s.textContent = t; box.appendChild(s); });
  }

  function setFrame(f) {
    state.frame = f; slider.value = f;
    timeEl.textContent = "t = " + D.times[f].toFixed(3) + " s (" + (f + 1) + " of " + nF + ")";
    render(true);
  }
  let raf = 0, last = 0, onScreen = true;
  function loop(ts) {
    if (!state.playing) return;
    const wait = state.frame === nF - 1 ? 1000 : 50; // about 20 frames per second, with a pause on the last frame
    if (onScreen && (!last || ts - last >= wait)) { last = ts; setFrame((state.frame + 1) % nF); }
    raf = requestAnimationFrame(loop);
  }
  function setPlaying(p) {
    state.playing = p; playBtn.textContent = p ? "Pause" : "Play";
    cancelAnimationFrame(raf);
    if (p) { last = 0; raf = requestAnimationFrame(loop); }
  }

  readTheme();
  views = progs.map(makeViewer).filter(Boolean);
  updateColorbar();
  setFrame(0);
  setPlaying(!matchMedia("(prefers-reduced-motion: reduce)").matches);

  slider.addEventListener("input", function () { setPlaying(false); setFrame(+slider.value); });
  playBtn.addEventListener("click", function () { setPlaying(!state.playing); });
  modeSel.addEventListener("change", function () { state.mode = modeSel.value; updateColorbar(); render(true); });
  resetBtn.addEventListener("click", function () { state.yaw = HOME.yaw; state.pitch = HOME.pitch; state.zoom = HOME.zoom; render(false); });
  if (window.ResizeObserver) new ResizeObserver(function () { render(false); }).observe(grid);
  else window.addEventListener("resize", function () { render(false); });
  if (window.IntersectionObserver) new IntersectionObserver(function (es) { onScreen = es[0].isIntersecting; }).observe(grid);
  const retheme = function () { readTheme(); updateColorbar(); render(true); };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", retheme);
  new MutationObserver(retheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
</script></body></html>
"""


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    if not any(os.path.isfile(os.path.join(results, f"run_summary_{k}.tsv")) for k in RUN_KINDS):
        sys.exit(f"No run summary in {results}/ - run src/cpu/run_all_cpu.sh or src/gpu/run_all_gpu.sh first.")
    path = os.path.join(results, "report.html")
    with open(path, "w") as f:
        f.write(build(results))
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
