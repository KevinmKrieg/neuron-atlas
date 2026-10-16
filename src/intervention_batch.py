"""Run baseline vs intervention generations for a small prompt set and emit Markdown."""
import argparse
import json
from pathlib import Path

from .common import load_model, MODEL_ID
from .intervene import generate_pair


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--hypotheses", default="data/hypotheses.example.json")
    p.add_argument("--prompts", default="data/intervention_prompts.jsonl")
    p.add_argument("--out", default="results/interventions.md")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--scale", type=float, default=0.0)
    a = p.parse_args()

    model, tokenizer, device = load_model(a.model, a.device)
    hypotheses = json.loads(Path(a.hypotheses).read_text())
    prompts = load_jsonl(a.prompts)

    sections = ["# Intervention examples", "", f"Scale factor: {a.scale}", ""]
    for hyp in hypotheses:
        sections += [f"## {hyp['name']} (layer {hyp['layer']}, neuron {hyp['neuron']}, {hyp.get('polarity','positive')})", ""]
        for row in prompts:
            if row.get("group") != hyp["label_field"]:
                continue
            baseline, modified = generate_pair(model, tokenizer, device, row["prompt"], hyp["layer"], hyp["neuron"], a.scale, a.max_new_tokens)
            sections += [f"### Prompt: {row['prompt']}", "", "**Baseline**", "", f"> {baseline}", "", "**Intervention**", "", f"> {modified}", ""]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
