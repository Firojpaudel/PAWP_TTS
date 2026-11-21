"""Grid-based generator for docs/figures/architecture.svg. Usage: python docs/figures/make_architecture.py docs/figures/architecture.svg"""
import sys

M, G, P = 40, 54, 18            # outer margin, gutter, inner padding
TOP, H = 120, 300               # main row top + height
CY = TOP + H // 2               # connector centerline
COLS = [("fe", 190), ("emb", 110), ("enc", 240), ("var", 232), ("aco", 186), ("dec", 232)]
X = {}
x = M
for k, w in COLS:
    X[k] = (x, w)
    x += w + G
W_CANVAS = x - G + M
BOT = TOP + H + 44              # bottom panels top
BH = 150
H_CANVAS = BOT + BH + 64

o = []
a = o.append
def t(x, y, s, cls="s", anchor="start", extra=""):
    a(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" {extra}>{s}</text>')
def box(x, y, w, h, fill, stroke, dash=False, shadow=True, r=14):
    d = ' stroke-dasharray="5 4"' if dash else ""
    f = ' filter="url(#sh)"' if shadow else ""
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"{d}{f}/>')
def stack(x, y, w, h, color):
    a(f'<rect x="{x+10}" y="{y+10}" width="{w}" height="{h}" rx="14" fill="{color}" opacity="0.35"/>')
    a(f'<rect x="{x+5}" y="{y+5}" width="{w}" height="{h}" rx="14" fill="{color}" opacity="0.6"/>')
def badge(x, y, label, color):
    a(f'<rect x="{x}" y="{y}" width="38" height="20" rx="10" fill="{color}"/>')
    a(f'<text x="{x+19}" y="{y+14}" text-anchor="middle" style="fill:#fff;font-size:12px;font-weight:800">{label}</text>')
def header(k, title, params, sub=None, bcol=None, blabel=None):
    x, w = X[k]
    t(x + P, TOP + 30, title, "bh")
    t(x + w - P, TOP + 30, params, "bp", "end")
    if sub:
        t(x + P, TOP + 50, sub, "xs")
    if blabel:
        badge(x + w - P - 38, TOP + 38, blabel, bcol)
def arrow(x1, x2, y=CY, label=None):
    a(f'<path d="M{x1},{y} L{x2},{y}" class="flow" marker-end="url(#ah)"/>')
    if label:
        t((x1 + x2) / 2, y - 10, label, "shape", "middle")

a(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_CANVAS} {H_CANVAS}" width="{W_CANVAS}" height="{H_CANVAS}" font-family="Inter, Segoe UI, Helvetica Neue, Arial, Noto Sans Devanagari, sans-serif">
<defs>
<marker id="ah" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker>
<linearGradient id="mel" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#312e81"/><stop offset="0.55" stop-color="#8b5cf6"/><stop offset="1" stop-color="#fde68a"/></linearGradient>
<filter id="sh" x="-10%" y="-10%" width="125%" height="130%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.09"/></filter>
<clipPath id="wc"><rect x="{X['dec'][0]+P}" y="{TOP+130}" width="{X['dec'][1]-2*P}" height="{H-130-P-28}" rx="8"/></clipPath>
<style>
.ttl{{font-size:24px;font-weight:750;fill:#0f172a;letter-spacing:-0.2px}}
.sub{{font-size:13.5px;fill:#64748b}}
.bh{{font-size:15.5px;font-weight:700;fill:#0f172a}}
.bp{{font-size:13px;font-weight:700;fill:#475569}}
.s{{font-size:13px;fill:#1e293b}}
.xs{{font-size:12px;fill:#334155}}
.lbl{{font-size:11px;font-weight:700;fill:#64748b;letter-spacing:1px}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;fill:#1e293b}}
.shape{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;fill:#0f172a;font-weight:600}}
.flow{{stroke:#475569;stroke-width:2;fill:none;stroke-dasharray:6 6;animation:dash 1.2s linear infinite}}
@keyframes dash{{to{{stroke-dashoffset:-24}}}}
.gEN{{animation:en 4s ease-in-out infinite}} .gNE{{animation:ne 4s ease-in-out infinite}}
@keyframes en{{0%,45%{{opacity:1}}50%,95%{{opacity:.18}}100%{{opacity:1}}}}
@keyframes ne{{0%,45%{{opacity:.18}}50%,95%{{opacity:1}}100%{{opacity:.18}}}}
.wave{{animation:wv 2.4s linear infinite}} @keyframes wv{{from{{transform:translateX(0)}}to{{transform:translateX(-40px)}}}}
@media (prefers-reduced-motion: reduce){{.flow,.gEN,.gNE,.wave{{animation:none}}}}
</style>
</defs>
<rect width="{W_CANVAS}" height="{H_CANVAS}" fill="#ffffff"/>''')

t(M, 50, "PAWP-TTS · compact bilingual (English–Nepali) speech synthesis", "ttl")
t(M, 76, "1,490,390 learned parameters in the complete inference graph · non-autoregressive · 10 ms lookahead · 24 kHz", "sub")

# ---------- column 1: input / frontend / tokens ----------
x, w = X["fe"]
t(x, TOP - 8, "INPUT", "lbl")
box(x, TOP, w, 58, "#f8fafc", "#e2e8f0", shadow=False, r=10)
t(x + P, TOP + 24, "आज मेरो office मा", "s", extra='font-weight="600" style="fill:#0f172a"')
t(x + P, TOP + 44, "meeting छ ।", "s", extra='font-weight="600" style="fill:#0f172a"')
a(f'<path d="M{x+w/2},{TOP+58} L{x+w/2},{TOP+86}" class="flow" marker-end="url(#ah)"/>')
fy = TOP + 86
box(x, fy, w, 118, "#ffffff", "#94a3b8", dash=True, shadow=False, r=12)
t(x + P, fy + 26, "Frontend", "bh")
t(x + P, fy + 44, "deterministic · 0 params", "xs")
a(f'<rect x="{x+P}" y="{fy+56}" width="70" height="24" rx="6" fill="#fff7ed" stroke="#fb923c"/>')
t(x + P + 35, fy + 72, "NE rules", "xs", "middle", 'style="fill:#9a3412"')
a(f'<rect x="{x+w-P-78}" y="{fy+56}" width="78" height="24" rx="6" fill="#eff6ff" stroke="#60a5fa"/>')
t(x + w - P - 39, fy + 72, "EN eSpeak", "xs", "middle", 'style="fill:#1e40af"')
t(x + P, fy + 100, "schwa deletion · स/श/ष→s", "xs")
ty = fy + 118 + 26
a(f'<path d="M{x+w/2},{fy+118} L{x+w/2},{ty-14}" class="flow" marker-end="url(#ah)"/>')
t(x, ty + 2, "PHONEME TOKENS", "lbl")
chips = [("a", "#fed7aa"), ("dz", "#fed7aa"), ("ɑ", "#bfdbfe"), ("f", "#bfdbfe"), ("ɪ", "#bfdbfe"), ("s", "#bfdbfe")]
for i, (c, col) in enumerate(chips):
    cx = x + i * 32
    a(f'<rect x="{cx}" y="{ty+10}" width="28" height="26" rx="6" fill="{col}"/>')
    a(f'<text x="{cx+14}" y="{ty+28}" text-anchor="middle" class="mono" style="fill:#0f172a">{c}</text>')
a(f'<rect x="{x}" y="{ty+46}" width="10" height="10" rx="2" fill="#fed7aa"/>')
t(x + 16, ty + 55, "Nepali", "xs")
a(f'<rect x="{x+70}" y="{ty+46}" width="10" height="10" rx="2" fill="#bfdbfe"/>')
t(x + 86, ty + 55, "English", "xs")
chip_y = ty + 23

# ---------- column 2: embed (centered on CY) ----------
x, w = X["emb"]
ey = CY - 50
box(x, ey, w, 100, "#f0f9ff", "#38bdf8")
t(x + w / 2, ey + 28, "Embed", "bh", "middle")
t(x + w / 2, ey + 48, "phoneme · lang", "xs", "middle")
t(x + w / 2, ey + 64, "stress · speaker", "xs", "middle")
t(x + w / 2, ey + 86, "18K", "bp", "middle")
# tokens -> embed (elbow into centerline)
fx = X["fe"][0] + X["fe"][1]
a(f'<path d="M{fx-2},{chip_y} L{fx+G/2},{chip_y} L{fx+G/2},{CY} L{x},{CY}" class="flow" marker-end="url(#ah)"/>')

# ---------- column 3: encoder ----------
x, w = X["enc"]
stack(x, TOP, w, H, "#bfdbfe")
box(x, TOP, w, H, "#eff6ff", "#3b82f6")
header("enc", "Text encoder", "367K", "ConvNeXt1D block", "#1d4ed8", "×4")
rows = [("DWConv k5 · lookahead 1", False), ("LayerNorm", False), ("PW 128→256 ⊕ ΔW_lang", True),
        ("GELU", False), ("PW 256→128 ⊕ ΔW_lang", True)]
ry = TOP + 66
iw = w - 2 * P - 14
for i, (s, hl) in enumerate(rows):
    yy = ry + i * 34
    a(f'<rect x="{x+P}" y="{yy}" width="{iw}" height="26" rx="6" fill="#fff" stroke="{"#3b82f6" if hl else "#93c5fd"}" stroke-width="{1.6 if hl else 1}"/>')
    t(x + P + iw / 2, yy + 17, s, "mono", "middle")
    if i:
        a(f'<path d="M{x+P+iw/2},{yy-8} L{x+P+iw/2},{yy}" stroke="#93c5fd" stroke-width="1.5"/>')
plus_y = ry + 5 * 34 + 8
a(f'<path d="M{x+P+iw/2},{plus_y-12} L{x+P+iw/2},{plus_y-10}" stroke="#93c5fd" stroke-width="1.5"/>')
a(f'<circle cx="{x+P+iw/2}" cy="{plus_y}" r="10" fill="#fff" stroke="#3b82f6"/>')
t(x + P + iw / 2, plus_y + 4.5, "+", "s", "middle", 'font-weight="700"')
rx_ = x + P + iw + 8
a(f'<path d="M{x+P+iw},{ry+13} L{rx_},{ry+13} L{rx_},{plus_y} L{x+P+iw/2+12},{plus_y}" stroke="#94a3b8" stroke-width="1.4" fill="none" marker-end="url(#ah)"/>')
t(x + P, TOP + H - P, "ΔW_lang: rank 16, per-token gate", "xs")

# ---------- column 4: variance adaptor ----------
x, w = X["var"]
box(x, TOP, w, H, "#f0fdf4", "#22c55e")
header("var", "Variance adaptor", "62K", "predicted at inference")
iw = w - 2 * P
yy = TOP + 64
a(f'<rect x="{x+P}" y="{yy}" width="{iw}" height="60" rx="8" fill="#fff" stroke="#86efac"/>')
t(x + P + 12, yy + 22, "Duration", "s", extra='font-weight="600" style="fill:#0f172a"')
t(x + P + 12, yy + 42, "frames / phoneme", "xs")
for i, hgt in enumerate([18, 30, 12, 34, 24, 16]):
    a(f'<rect x="{x+P+iw-10-6*9+i*9}" y="{yy+48-hgt}" width="7" height="{hgt}" rx="2" fill="#22c55e"/>')
yy += 70
a(f'<rect x="{x+P}" y="{yy}" width="{iw}" height="60" rx="8" fill="#fff" stroke="#86efac"/>')
t(x + P + 12, yy + 22, "Length regulator", "s", extra='font-weight="600" style="fill:#0f172a"')
bx = x + P + 12
a(f'<rect x="{bx}" y="{yy+32}" width="14" height="18" rx="3" fill="#fed7aa"/><rect x="{bx+17}" y="{yy+32}" width="14" height="18" rx="3" fill="#bfdbfe"/>')
a(f'<path d="M{bx+36},{yy+41} L{bx+54},{yy+41}" stroke="#64748b" stroke-width="1.5" marker-end="url(#ah)"/>')
for i in range(8):
    col = "#fed7aa" if i < 3 else "#bfdbfe"
    a(f'<rect x="{bx+60+i*9}" y="{yy+32}" width="7" height="18" rx="1.5" fill="{col}"/>')
t(x + P + iw - 12, yy + 46, "L → T", "shape", "end")
yy += 70
hw = (iw - 10) / 2
for j, (name, path) in enumerate([("Pitch F0", "c14,-18 24,6 36,-12 s22,-10 36,6"), ("Energy", "l12,-16 l12,8 l12,-16 l12,12 l12,-8 l12,16")]):
    bx = x + P + j * (hw + 10)
    a(f'<rect x="{bx}" y="{yy}" width="{hw}" height="64" rx="8" fill="#fff" stroke="#86efac"/>')
    t(bx + 10, yy + 20, name, "s", extra='font-weight="600" style="fill:#0f172a"')
    a(f'<path d="M{bx+10},{yy+50} {path}" stroke="#16a34a" stroke-width="2" fill="none"/>')
t(x + P, TOP + H - P, "trained on CTC durations", "xs")

# ---------- column 5: acoustic ----------
x, w = X["aco"]
stack(x, TOP, w, H, "#ddd6fe")
box(x, TOP, w, H, "#f5f3ff", "#8b5cf6")
header("aco", "Acoustic net", "378K", "ConvNeXt1D k=7", "#6d28d9", "×4")
t(x + P, TOP + 72, "+ language residuals", "xs")
t(x + P, TOP + 90, "→ Linear 128→80", "xs")
mx, my, mw, mh = x + P, TOP + 106, w - 2 * P, H - 106 - 46
a(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="8" fill="#1e1b4b"/>')
hs = [0.45, 0.6, 0.85, 0.7, 0.35, 0.75, 0.9, 0.55, 0.3, 0.7, 0.82, 0.5, 0.4, 0.62]
bw = (mw - 12) / len(hs)
for i, hh in enumerate(hs):
    bh = (mh - 12) * hh
    a(f'<rect x="{mx+6+i*bw+1}" y="{my+mh-6-bh}" width="{bw-2}" height="{bh}" fill="url(#mel)"/>')
t(x + w / 2, TOP + H - P, "log-mel  T×80", "shape", "middle")

# ---------- column 6: decoder + waveform ----------
x, w = X["dec"]
stack(x, TOP, w, H, "#fbcfe8")
box(x, TOP, w, H, "#fdf2f8", "#ec4899")
header("dec", "Waveform decoder", "665K", "Vocos-style ConvNeXt1D", "#be185d", "×3")
t(x + P, TOP + 76, "→ |X| + phase · 513 bins", "mono")
t(x + P, TOP + 96, "→ iSTFT 1024 / hop 240", "mono")
wy = TOP + 130
a(f'<rect x="{x+P}" y="{wy}" width="{w-2*P}" height="{H-130-P-28}" rx="8" fill="#0f172a"/>')
seg = "q5,-34 10,0 t10,0 q5,-48 10,0 t10,0 q5,-20 10,0 t10,0 q5,-40 10,0 t10,0"
a(f'<g clip-path="url(#wc)"><path class="wave" stroke="#f9a8d4" stroke-width="1.8" fill="none" d="M{x+P},{wy+(H-130-P-28)/2} {" ".join([seg]*6)}"/></g>')
t(x + w / 2, TOP + H - P, "24 kHz audio · chunked", "shape", "middle")

# main connectors on the centerline (start after stacked-card offsets)
arrow(X["emb"][0] + X["emb"][1], X["enc"][0], label="L×128")
arrow(X["enc"][0] + X["enc"][1] + 10, X["var"][0], label="L×128")
arrow(X["var"][0] + X["var"][1], X["aco"][0], label="T×128")
arrow(X["aco"][0] + X["aco"][1] + 10, X["dec"][0], label="T×80")

# ---------- bottom row ----------
x, w = X["enc"]
box(x, BOT, w, BH, "#fffbeb", "#f59e0b", shadow=False)
t(x + P, BOT + 28, "Language-gated residual", "bh")
t(x + P, BOT + 52, "y = W·x", "mono")
for k, (cls, fill, stroke, s) in enumerate([("gEN", "#dbeafe", "#3b82f6", "+ m_en(t) · U_en V_en x"), ("gNE", "#ffedd5", "#f97316", "+ m_ne(t) · U_ne V_ne x")]):
    yy = BOT + 62 + k * 30
    a(f'<g class="{cls}"><rect x="{x+P}" y="{yy}" width="{w-2*P}" height="24" rx="6" fill="{fill}" stroke="{stroke}"/>')
    a(f'<text x="{x+w/2}" y="{yy+16}" class="mono" text-anchor="middle">{s}</text></g>')
t(x + P, BOT + BH - P + 2, "switches mid-sentence · foldable into W", "xs")
a(f'<path d="M{x+w/2},{TOP+H+12} L{x+w/2},{BOT}" stroke="#f59e0b" stroke-width="1.6" stroke-dasharray="3 4"/>')

x = X["var"][0]
w = X["aco"][0] + X["aco"][1] - x
box(x, BOT, w, BH, "#f8fafc", "#cbd5e1", dash=True, shadow=False)
t(x + P, BOT + 28, "Training only · not in the deployed graph", "bh")
for k, s in enumerate(["CTC phoneme recognizer → Viterbi forced-alignment durations",
                       "Multi-period + multi-resolution discriminators (GAN)",
                       "Speaker table (1,814 voices) → one voice kept at export",
                       "Loss: L1 log-mel + MR-STFT + dur/F0/energy + adv + FM"]):
    yy = BOT + 54 + k * 24
    a(f'<circle cx="{x+P+3}" cy="{yy-4}" r="3" fill="#64748b"/>')
    t(x + P + 14, yy, s, "s")

x, w = X["dec"]
t(x, BOT + 28, "Parameter budget", "bh")
t(x, BOT + 46, "share of 1,490,390", "xs")
parts = [(0.0124, "#38bdf8", "Embeddings 1.2%"), (0.246, "#3b82f6", "Encoder 24.6%"), (0.0418, "#22c55e", "Variance 4.2%"),
         (0.2538, "#8b5cf6", "Acoustic 25.4%"), (0.446, "#ec4899", "Decoder 44.6%")]
cx = x
for f, col, _ in parts:
    a(f'<rect x="{cx}" y="{BOT+58}" width="{w*f}" height="22" fill="{col}"/>')
    cx += w * f
order = [parts[1], parts[3], parts[4], parts[2], parts[0]]
for k, (_, col, lab) in enumerate(order):
    lx = x + (k % 2) * (w / 2)
    ly = BOT + 98 + (k // 2) * 20
    a(f'<rect x="{lx}" y="{ly-9}" width="10" height="10" rx="2" fill="{col}"/>')
    t(lx + 16, ly, lab, "xs")

t(M, H_CANVAS - 26, "L = phoneme tokens · T = 10 ms frames · counts from configs/student_1_5m.yaml; every learned tensor used at inference is included", "xs")
a("</svg>")
open(sys.argv[1], "w").write("\n".join(o))
print(W_CANVAS, H_CANVAS)
