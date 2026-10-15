"""Display strongest neurons at one layer for human inspection."""
import argparse
import json
from pathlib import Path


def render(data, layer, count=30):
    neurons = [n for n in data["neurons"] if n["layer"] == layer]
    neurons.sort(key=lambda n: max((abs(e["activation"]) for e in n["examples"]),
                                   default=0), reverse=True)
    lines = [f"# Native neuron candidates — layer {layer}", "",
             "Highest magnitudes, not interpretability rankings. Inspect positive and negative poles separately.", ""]
    for n in neurons[:count]:
        lines += [f'## Neuron {n["neuron"]}', ""]
        for e in n["examples"]:
            lines.append(f'- activation `{e["activation"]:+.3f}`; token `{e["token"]}`; context: {e["context"]!r}')
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", default="results/scan.json")
    p.add_argument("--out", default="results/report.md")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--count", type=int, default=30)
    a = p.parse_args()
    output = render(json.loads(Path(a.scan).read_text()), a.layer, a.count)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(output)
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
