"""Generate the paper's charts as clean vector SVGs (stdlib only); render to PDF via headless Chrome.

Usage:  make -C paper figures   (writes paper/figures/*.pdf)
All numbers below are copied from docs/EXPERIMENTS.md / artifacts/reports (see per-figure notes).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Reference palette (validated categorical slots 1-3, fixed order) + neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, GRAY_L = "#898781", "#c3c2b7"
INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e7e6e0", "#c3c2b7"
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
W1 = 340  # single-column figure width (px)
TEXT_SCALE = 1.55  # keeps printed labels >= ~7 pt at column width


def svg(w: int, h: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{FONT}">\n<rect width="{w}" height="{h}" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def t(x: float, y: float, s: str, size: float = 8.5, fill: str = INK, anchor: str = "start",
      weight: str = "normal", extra: str = "") -> str:
    size = round(size * TEXT_SCALE, 2)
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {extra}>{s}</text>')


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, w: float = 0.8,
         dash: str = "") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{w}"{d}/>')


def rbar_h(x: float, y: float, w: float, h: float, fill: str, r: float = 2.0) -> str:
    """Horizontal bar anchored at x (baseline) with rounded data-end on the right."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" '
            f'fill="{fill}"/>')


def rbar_v(x: float, base: float, w: float, h: float, fill: str, r: float = 2.0) -> str:
    """Vertical bar anchored at the baseline with rounded data-end on top."""
    if h <= 0:
        return ""
    r = min(r, h, w / 2)
    top = base - h
    return (f'<path d="M{x:.1f},{base:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{top:.1f} {x + w:.1f},{top + r:.1f} V{base:.1f} Z" '
            f'fill="{fill}"/>')


def legend(x: float, y: float, items: list[tuple[str, str]], gap: float = 14) -> list[str]:
    out, cx = [], x
    for label, col in items:
        out.append(f'<rect x="{cx:.1f}" y="{y - 6.5:.1f}" width="8" height="8" rx="2" fill="{col}"/>')
        out.append(t(cx + 11, y, label, 8, INK2))
        cx += 11 + len(label) * 4.45 * TEXT_SCALE + gap
    return out


# ------------------------------------------------------------------ Fig: label consistency
def fig_label_consistency() -> str:
    # EXP-014 / EXP-015a / EXP-013a: v2 (stale labels) -> v3 (consistent labels).
    # Two panels with their own scales (PER ~4-7 % vs व error rate 0-55 %).
    panels = [
        ("a  Phoneme error rate, dev (%)", [("Nepali", 7.06, 3.96), ("English", 5.29, 4.33)], 8.0, 2),
        ("b  व-word /b/ heard as /w/, test (%)", [("व words", 55.1, 0.0)], 60.0, 10),
    ]
    w, h = W1, 194
    x0, x1 = 98, w - 36
    b: list[str] = []
    b += legend(x0, 16, [("v2 (stale labels)", GRAY), ("v3 (consistent)", BLUE)], gap=12)
    y = 38
    for title, rows, vmax, tick in panels:
        def sx(v: float, vmax: float = vmax) -> float:
            return x0 + (x1 - x0) * v / vmax
        b.append(t(x0 - 58, y + 2, title, 8.0, INK, "start", "bold"))
        top = y + 12
        bot = top + len(rows) * 20 + 2
        k = 0
        while k <= vmax + 1e-9:
            b.append(line(sx(k), top - 2, sx(k), bot, GRID, 0.8))
            b.append(t(sx(k), bot + 10, f"{k:g}", 7.2, GRAY, "middle"))
            k += tick
        for i, (lab, v2, v3) in enumerate(rows):
            yy = top + 9 + i * 20
            b.append(t(x0 - 8, yy + 3, lab, 8.2, INK, "end"))
            b.append(line(sx(v3), yy, sx(v2), yy, GRAY_L, 2.2))
            b.append(f'<circle cx="{sx(v2):.1f}" cy="{yy}" r="4" fill="{GRAY}" stroke="#fff" stroke-width="1.4"/>')
            b.append(f'<circle cx="{sx(v3):.1f}" cy="{yy}" r="4" fill="{BLUE}" stroke="#fff" stroke-width="1.4"/>')
            b.append(t(sx(v2) + 7, yy + 3, f"{v2:g}", 7.4, INK2))
            b.append(t(sx(v3) - 7, yy + 3, f"{v3:g}", 7.4, BLUE, "end", "bold") if v3 > 0.5
                     else t(sx(v3) + 7, yy + 3, f"{v3:g}", 7.4, BLUE, "start", "bold"))
        y = bot + 26
    return svg(w, h, b)


# ------------------------------------------------------------------ Fig: intelligibility bars
def fig_intelligibility() -> str:
    # Table 1 (EXP-015a/b/c): held-out ASR CER %, one recognizer, n = 30/30/9
    groups = ["English", "Nepali", "Code-mixed"]
    series = [
        ("Human recordings", GRAY_L, [0.75, 3.82, 12.88]),
        ("PAWP-1.5M", BLUE, [1.50, 1.75, 5.32]),
        ("281K shared", ORANGE, [2.93, 5.93, 5.01]),
        ("281K dense", AQUA, [2.42, 6.59, 9.01]),
    ]
    w, h = W1, 202
    left, right, top, base = 40, w - 8, 58, 172
    vmax = 14.0

    def sy(v: float) -> float:
        return base - (base - top) * v / vmax

    b: list[str] = []
    for k in range(0, 15, 2):
        b.append(line(left, sy(k), right, sy(k), GRID if k else BASE, 0.8 if k else 1.0))
        b.append(t(left - 5, sy(k) + 3, f"{k}", 7.5, GRAY, "end"))
    b.append(t(left - 22, top - 10, "CER (%)", 7.5, GRAY))
    gw = (right - left) / 3
    bw, gap = 15, 2
    for gi, g in enumerate(groups):
        gx = left + gi * gw + (gw - (4 * bw + 3 * gap)) / 2
        for si, (_, col, vals) in enumerate(series):
            v = vals[gi]
            x = gx + si * (bw + gap)
            b.append(rbar_v(x, base, bw, base - sy(v), col))
            b.append(t(x + bw / 2, sy(v) - 3, f"{v:.1f}", 6.6, INK2, "middle"))
        b.append(t(left + gi * gw + gw / 2, base + 13, g, 8.2, INK, "middle"))
    b += legend(left - 4, 12, [(s[0], s[1]) for s in series[:2]], gap=14)
    b += legend(left - 4, 30, [(s[0], s[1]) for s in series[2:]], gap=14)
    return svg(w, h, b)


# ------------------------------------------------------------------ Fig: shared vs dense
def fig_shared_dense() -> str:
    # EXP-015b/c, artifacts/reports/exp015bc_multiseed_test300.json: test PER (shared - dense), pp,
    # n = 300 NE + 300 EN untouched test, seeds 42 / 101 / 202.
    rows = [("Nepali PER", [0.744, 0.843, 0.877]), ("English PER", [1.848, 0.788, 1.043])]
    w, h = W1, 118
    x0, x1 = 118, w - 66
    lo, hi = -0.5, 2.5

    def sx(v: float) -> float:
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    b: list[str] = []
    top = 26
    for k in [0, 1, 2]:
        b.append(line(sx(k), top - 8, sx(k), top + 50, BASE if k == 0 else GRID, 1.0 if k == 0 else 0.8))
        b.append(t(sx(k), top + 62, f"{k:+d}" if k else "0", 7.5, GRAY, "middle"))
    b += legend(18, 12, [("each seed", GRAY), ("mean of 3 seeds (tick)", ORANGE)], gap=12)
    for i, (lab, vals) in enumerate(rows):
        y = top + 10 + i * 26
        m = sum(vals) / len(vals)
        b.append(t(x0 - 8, y + 3, lab, 8.2, INK, "end"))
        b.append(line(sx(min(vals)), y, sx(max(vals)), y, GRAY_L, 2.0))
        b.append(line(sx(m), y - 8, sx(m), y + 8, ORANGE, 2.6))
        for v in vals:
            b.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="3.4" fill="{GRAY}" stroke="#fff" stroke-width="1.2"/>')
        b.append(t(sx(max(vals)) + 7, y + 3, f"+{m:.2f} pp", 7.5, INK2))
    b.append(t(sx(0) - 4, top + 78, "← shared better", 7.2, GRAY, "end"))
    b.append(t(sx(0) + 4, top + 78, "dense better →", 7.2, GRAY))
    return svg(w, h + 10, b)


# ------------------------------------------------------------------ Fig: WSP forest plot
def fig_wsp_forest() -> str:
    # Δ Nepali PER (edit − base, percentage points) with paired-bootstrap 95% CI, n = 300 NE test.
    # EXP-013a / EXP-016 controls; 1-parameter calibration selected on val, tested once.
    rows = [
        ("EXP-013a  rank-1 edit (shared layer)", BLUE, 1.22, 1.11, 2.19),
        ("random, same location & norm", GRAY, 0.02, 0.001, 0.12),
        ("gradient FT, same parameters", ORANGE, 0.02, -0.38, 0.025),
        (None, None, 0, 0, 0),
        ("EXP-016  96-param pause edit", BLUE, -0.11, -0.34, 0.12),
        ("random, same location & norm", GRAY, -0.20, -0.39, -0.06),
        ("gradient FT, same parameters", ORANGE, 0.51, 0.22, 0.82),
        (None, None, 0, 0, 0),
        ("1-parameter duration calibration", AQUA, -0.271, -0.414, -0.118),
    ]
    w, h = 600, 208
    x0, x1 = 286, w - 74
    lo, hi = -0.6, 2.4

    def sx(v: float) -> float:
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    b: list[str] = []
    top, step = 26, 15
    n_rows = len(rows)
    bottom = top + n_rows * step + 4
    for k in [-0.5, 0, 0.5, 1.0, 1.5, 2.0]:
        b.append(line(sx(k), top - 2, sx(k), bottom, BASE if k == 0 else GRID, 1.0 if k == 0 else 0.8))
        b.append(t(sx(k), bottom + 11, f"{k:+.1f}" if k else "0", 7.5, GRAY, "middle"))
    b.append(t((x0 + x1) / 2, bottom + 24, "Δ Nepali PER vs. baseline (percentage points, 95% CI)", 7.8, INK2, "middle"))
    b.append(t(sx(0) - 4, top - 5, "← better", 7.2, GRAY, "end"))
    b.append(t(sx(0) + 4, top - 5, "worse →", 7.2, GRAY))
    for i, (lab, col, m, a, c) in enumerate(rows):
        if lab is None:
            continue
        y = top + 6 + i * step
        bold = "bold" if col in (BLUE, AQUA) else "normal"
        indent = 0 if col in (BLUE, AQUA) else 10
        b.append(t(x0 - 10 - 0 * indent, y + 3, lab, 7.9, INK if bold == "bold" else INK2, "end", bold))
        b.append(line(sx(a), y, sx(c), y, col, 1.8))
        b.append(f'<circle cx="{sx(m):.1f}" cy="{y}" r="3.6" fill="{col}" stroke="#fff" stroke-width="1.3"/>')
        b.append(t(x1 + 8, y + 3, f"{m:+.2f}", 7.5, INK2))
    for i in (3, 7):
        y = top + 6 + i * step
        b.append(line(18, y, w - 12, y, GRID, 0.6, "2 3"))
    return svg(w, h, b)


# ------------------------------------------------------------------ Fig: duration bias
def fig_duration_bias() -> str:
    # audit_003_diagnostics.json: item3_v3_val_ne_dur_bias_by_class (pred / CTC ground truth)
    rows = [("Fricatives", 0.965), ("Nasals", 0.943), ("Stops", 0.938), ("Vowels", 0.928),
            ("Liquids / approx.", 0.903), ("Other", 0.869)]
    w, h = W1, 170
    x0, x1 = 132, w - 46
    lo, hi = 0.80, 1.02

    def sx(v: float) -> float:
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    b: list[str] = []
    top, step = 22, 18
    bottom = top + len(rows) * step
    for k in [0.80, 0.85, 0.90, 0.95, 1.00]:
        b.append(line(sx(k), top - 2, sx(k), bottom, BASE if k == 1.0 else GRID, 1.2 if k == 1.0 else 0.8,
                      "3 2" if k == 1.0 else ""))
        b.append(t(sx(k), bottom + 11, f"{k:.2f}", 7.5, GRAY, "middle"))
    b.append(t(sx(1.0), top - 6, "unbiased", 7.2, GRAY, "end"))
    for i, (lab, r) in enumerate(rows):
        y = top + i * step + 3
        b.append(t(x0 - 8, y + 8, lab, 8.2, INK, "end"))
        b.append(rbar_h(sx(lo), y, sx(r) - sx(lo), 11, BLUE))
        b.append(t(sx(r) + 5, y + 8.5, f"{r:.3f}", 7.5, INK2))
    b.append(t((x0 + x1) / 2, bottom + 24, "predicted / aligned duration (NE val)", 7.8, INK2, "middle"))
    return svg(w, h, b)


# ------------------------------------------------------------------ Fig: pause selectivity
def fig_pause_selectivity() -> str:
    # EXP-016: mean duration shift (frames) on <sp> word-boundary tokens vs regular phonemes
    groups = [("Pause edit", 1.001, 0.037), ("Random edit", 0.129, 0.092),
              ("Gradient FT", -0.702, -0.362)]
    w, h = W1, 150
    x0, x1 = 122, w - 46
    lo, hi = -0.9, 1.2

    def sx(v: float) -> float:
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    b: list[str] = []
    top, gstep, bh = 24, 34, 10
    bottom = top + len(groups) * gstep - 4
    for k in [-0.5, 0, 0.5, 1.0]:
        b.append(line(sx(k), top - 4, sx(k), bottom, BASE if k == 0 else GRID, 1.0 if k == 0 else 0.8))
        b.append(t(sx(k), bottom + 11, f"{k:+.1f}" if k else "0", 7.5, GRAY, "middle"))
    for i, (lab, sp, ph) in enumerate(groups):
        y = top + i * gstep
        b.append(t(x0 - 8, y + 12, lab, 8.0, INK, "end"))
        for j, (v, col) in enumerate([(sp, BLUE), (ph, GRAY_L)]):
            yy = y + j * (bh + 2)
            if v >= 0:
                b.append(rbar_h(sx(0), yy, sx(v) - sx(0), bh, col))
                b.append(t(sx(v) + 4, yy + 8, f"{v:+.2f}", 7.2, INK2))
            else:
                x = sx(v)
                b.append(f'<path d="M{sx(0):.1f},{yy} H{x + 2:.1f} Q{x:.1f},{yy} {x:.1f},{yy + 2} V{yy + bh - 2} '
                         f'Q{x:.1f},{yy + bh} {x + 2:.1f},{yy + bh} H{sx(0):.1f} Z" fill="{col}"/>')
                ink = "#ffffff" if col == BLUE else INK
                b.append(t(x + 4, yy + 8, f"{v:+.2f}", 7.2, ink, "start", "bold" if col == BLUE else "normal"))
    b += legend(18, 12, [("word-boundary pause ⟨sp⟩", BLUE), ("regular phonemes", GRAY_L)], gap=10)
    b.append(t((x0 + x1) / 2, bottom + 24, "mean duration change (frames, 10 ms each)", 7.8, INK2, "middle"))
    return svg(w, h, b)


def to_pdf(svg_path: Path) -> None:
    import re

    txt = svg_path.read_text(encoding="utf-8")
    m = re.search(r'width="(\d+(?:\.\d+)?)" height="(\d+(?:\.\d+)?)"', txt)
    assert m
    w, h = float(m.group(1)), float(m.group(2))
    html = (f"<!doctype html><html><head><meta charset='utf-8'><style>@page{{size:{w}px {h}px;margin:0}}"
            f"html,body{{margin:0;padding:0}}svg{{display:block}}</style></head><body>{txt}</body></html>")
    with tempfile.TemporaryDirectory() as td:
        hp = Path(td) / "f.html"
        hp.write_text(html, encoding="utf-8")
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--no-pdf-header-footer", f"--print-to-pdf={svg_path.with_suffix('.pdf')}",
                        hp.as_uri()], check=True, capture_output=True, timeout=120)


def main() -> None:
    figs = {
        "label_consistency": fig_label_consistency(),
        "intelligibility": fig_intelligibility(),
        "shared_vs_dense": fig_shared_dense(),
        "wsp_forest": fig_wsp_forest(),
        "duration_bias": fig_duration_bias(),
        "pause_selectivity": fig_pause_selectivity(),
    }
    with tempfile.TemporaryDirectory() as td:  # SVGs are intermediates; only PDFs are kept
        for name, s in figs.items():
            p = Path(td) / f"{name}.svg"
            p.write_text(s, encoding="utf-8")
            to_pdf(p)
            (OUT / f"{name}.pdf").write_bytes(p.with_suffix(".pdf").read_bytes())
            print("wrote", f"{name}.pdf")


if __name__ == "__main__":
    main()
