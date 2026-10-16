"""Generate a polished SVG summary of native neuron scan results."""
import argparse
import html
import json
from pathlib import Path

BG = "#f7f5ef"
CARD = "#ffffff"
BORDER = "#ddd7ca"
TEXT = "#18181b"
MUTED = "#66615a"
ACCENT = "#2251d1"
POS = "#2f6fed"
NEG = "#c46b00"
CHIP_BG = "#f1ede4"
GRID = "#ece7db"


def esc(text):
    return html.escape(str(text))


def wrap(text, width=46):
    words = str(text).split()
    lines, line = [], ""
    for word in words:
        proposed = (line + " " + word).strip()
        if len(proposed) <= width:
            line = proposed
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def top_neurons(data, layer, count=8):
    neurons = [n for n in data["neurons"] if n["layer"] == layer]
    neurons.sort(key=lambda n: max((abs(e["activation"]) for e in n["examples"]), default=0), reverse=True)
    return neurons[:count]


def stats(data, layer):
    neurons = [n for n in data["neurons"] if n["layer"] == layer]
    layers = sorted(set(data.get("layers", [])))
    max_abs = max((abs(e["activation"]) for n in neurons for e in n["examples"]), default=0)
    return {
        "neurons": len(neurons),
        "lines": data.get("num_lines", 0),
        "model": data.get("model", "unknown"),
        "top_k": data.get("top_k", 0),
        "layers": ", ".join(map(str, layers)),
        "max_abs": max_abs,
    }


def card(svg, x, y, w, h, neuron):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{CARD}" stroke="{BORDER}"/>')
    examples = neuron["examples"]
    strongest = max(examples, key=lambda e: abs(e["activation"]))
    dom = strongest["activation"]
    dom_color = POS if dom >= 0 else NEG
    polarity = "positive pole" if dom >= 0 else "negative pole"
    svg.append(f'<text x="{x+22}" y="{y+34}" font-size="20" font-weight="700" fill="{TEXT}">Neuron {neuron["neuron"]}</text>')
    svg.append(f'<text x="{x+22}" y="{y+58}" font-size="12" fill="{MUTED}">Layer {neuron["layer"]} · strongest {polarity}</text>')
    chip_w = 118
    svg.append(f'<rect x="{x+w-chip_w-18}" y="{y+18}" width="{chip_w}" height="30" rx="15" fill="{CHIP_BG}"/>')
    svg.append(f'<text x="{x+w-chip_w/2-18}" y="{y+38}" text-anchor="middle" font-size="13" font-weight="700" fill="{dom_color}">{dom:+.2f}</text>')

    yy = y + 86
    for ex in examples[:3]:
        color = POS if ex["activation"] >= 0 else NEG
        svg.append(f'<rect x="{x+22}" y="{yy-15}" width="62" height="24" rx="12" fill="{CHIP_BG}"/>')
        svg.append(f'<text x="{x+53}" y="{yy+1}" text-anchor="middle" font-size="12" font-weight="700" fill="{color}">{ex["activation"]:+.2f}</text>')
        token = esc(ex["token"]).replace(" ", "␠")
        svg.append(f'<text x="{x+95}" y="{yy+1}" font-size="12" font-weight="700" fill="{TEXT}">token {token}</text>')
        yy += 22
        for i, line in enumerate(wrap(ex["context"], 50)[:2]):
            svg.append(f'<text x="{x+95}" y="{yy+1}" font-size="11" fill="{MUTED}">{esc(line)}</text>')
            yy += 15
        yy += 10


def render_svg(data, layer, title="Neuron Atlas", subtitle=None):
    top = top_neurons(data, layer, count=8)
    s = stats(data, layer)
    width = 1600
    height = 1400
    if subtitle is None:
        subtitle = "Native MLP neurons in Qwen3-0.6B — strongest activation contexts from a cloud scan"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        '<defs><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000" flood-opacity="0.05"/></filter></defs>',
    ]

    svg.append(f'<text x="80" y="94" font-size="44" font-weight="800" fill="{TEXT}">{esc(title)}</text>')
    svg.append(f'<text x="80" y="128" font-size="18" fill="{MUTED}">{esc(subtitle)}</text>')
    svg.append(f'<text x="80" y="160" font-size="14" fill="{MUTED}">Layer {layer} summary · inspect positive and negative poles separately</text>')

    chip_y = 190
    chips = [
        ("model", s["model"]),
        ("layer", str(layer)),
        ("corpus lines", str(s["lines"])),
        ("neurons scanned", str(s["neurons"])),
        ("top-k examples", str(s["top_k"])),
        ("layers in scan", s["layers"]),
    ]
    x = 80
    for label, value in chips:
        w = max(140, 70 + 8 * len(value))
        svg.append(f'<rect x="{x}" y="{chip_y}" width="{w}" height="46" rx="18" fill="{CARD}" stroke="{BORDER}"/>')
        svg.append(f'<text x="{x+18}" y="{chip_y+19}" font-size="12" fill="{MUTED}">{esc(label)}</text>')
        svg.append(f'<text x="{x+18}" y="{chip_y+35}" font-size="15" font-weight="700" fill="{TEXT}">{esc(value)}</text>')
        x += w + 14

    # Summary chart
    panel_x, panel_y, panel_w, panel_h = 80, 272, 1440, 340
    svg.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="22" fill="{CARD}" stroke="{BORDER}" filter="url(#shadow)"/>')
    svg.append(f'<text x="{panel_x+28}" y="{panel_y+38}" font-size="22" font-weight="800" fill="{TEXT}">Top neurons by absolute activation magnitude</text>')
    svg.append(f'<text x="{panel_x+28}" y="{panel_y+62}" font-size="13" fill="{MUTED}">Bar length encodes strongest |activation| among the saved top contexts. Color shows dominant polarity.</text>')

    chart_x = panel_x + 26
    chart_y = panel_y + 92
    chart_w = panel_w - 52
    max_val = max((max(abs(e["activation"]) for e in n["examples"]) for n in top), default=1.0)
    bar_area = chart_w - 500
    row_h = 18
    gap = 10
    for i, n in enumerate(top[:8]):
        yy = chart_y + i * (row_h + gap)
        max_ex = max(n["examples"], key=lambda e: abs(e["activation"]))
        value = abs(max_ex["activation"])
        color = POS if max_ex["activation"] >= 0 else NEG
        bar_w = max(2, (value / max_val) * bar_area)
        svg.append(f'<text x="{chart_x}" y="{yy+13}" font-size="12" font-weight="700" fill="{TEXT}">N{n["neuron"]}</text>')
        svg.append(f'<rect x="{chart_x+64}" y="{yy}" width="{bar_area}" height="{row_h}" rx="9" fill="{GRID}"/>')
        svg.append(f'<rect x="{chart_x+64}" y="{yy}" width="{bar_w}" height="{row_h}" rx="9" fill="{color}"/>')
        svg.append(f'<text x="{chart_x+74+bar_area}" y="{yy+13}" font-size="12" fill="{MUTED}">{value:.2f}</text>')
        context = esc(max_ex["context"][:52] + ("…" if len(max_ex["context"]) > 52 else ""))
        svg.append(f'<text x="{chart_x+64+bar_area+70}" y="{yy+13}" font-size="12" fill="{MUTED}">{context}</text>')

    # cards
    start_x, start_y = 80, 630
    card_w, card_h = 690, 300
    x_positions = [80, 830]
    idx = 0
    for row in range(2):
        for col in range(2):
            if idx >= min(4, len(top)):
                break
            card(svg, x_positions[col], start_y + row * (card_h + 28), card_w, card_h, top[idx])
            idx += 1

    # footer note
    note_y = 1350
    note = "Interpretability is not established by activation alone. Use held-out validation, controls, and interventions before treating any neuron explanation as real."
    svg.append(f'<text x="80" y="{note_y}" font-size="13" fill="{MUTED}">{esc(note)}</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", default="results/scan.json")
    p.add_argument("--out", default="results/scan.svg")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--title", default="Neuron Atlas")
    p.add_argument("--subtitle", default=None)
    a = p.parse_args()
    data = json.loads(Path(a.scan).read_text())
    svg = render_svg(data, a.layer, a.title, a.subtitle)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
