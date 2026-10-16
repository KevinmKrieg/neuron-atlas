"""Validate neuron hypotheses on held-out labeled text with AUROC and random controls."""
import argparse
import json
import random
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-0.6B"


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def auc_roc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def pick_controls(total_neurons, avoid, k, rng):
    pool = [i for i in range(total_neurons) if i not in avoid]
    rng.shuffle(pool)
    return pool[:k]


def score_specs(model, tokenizer, device, texts, specs, max_tokens=128):
    from .common import down_projections
    projections = down_projections(model)
    grouped = {}
    for idx, spec in enumerate(specs):
        grouped.setdefault(spec["layer"], []).append((idx, spec))
    store = {}
    handles = []

    def make_hook(layer):
        idx_specs = grouped[layer]
        def hook(module, inputs):
            x = inputs[0].detach().float()[0].cpu()
            for idx, spec in idx_specs:
                col = x[:, spec["neuron"]]
                sign = 1.0 if spec.get("polarity", "positive") == "positive" else -1.0
                score = float(torch.max(col * sign).item())
                store[idx] = score
        return hook

    for layer in grouped:
        handles.append(projections[layer].register_forward_pre_hook(make_hook(layer)))

    outputs = []
    try:
        with torch.inference_mode():
            for text in texts:
                batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens, add_special_tokens=False)
                store.clear()
                model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
                outputs.append([store.get(i) for i in range(len(specs))])
    finally:
        for h in handles:
            h.remove()
    return outputs


def render_markdown(results, examples_count, dataset_name):
    lines = [f"# Held-out neuron validation — {dataset_name}", "", f"Examples: {examples_count}", "",
             "| Hypothesis | Layer | Neuron | Pole | Label | Positives | AUROC | Mean control AUROC | Delta |", 
             "|---|---:|---:|---|---|---:|---:|---:|---:|"]
    for r in results:
        auc = "n/a" if r["auroc"] is None else f'{r["auroc"]:.3f}'
        c = "n/a" if r["control_mean_auroc"] is None else f'{r["control_mean_auroc"]:.3f}'
        d = "n/a" if r["delta_vs_control"] is None else f'{r["delta_vs_control"]:+.3f}'
        lines.append(f'| {r["name"]} | {r["layer"]} | {r["neuron"]} | {r["polarity"]} | {r["label_field"]} | {r["positives"]} | {auc} | {c} | {d} |')
    lines += ["", "AUROC is computed from the maximum pole-aligned activation of each neuron across the tokens in a sentence."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--dataset", default="data/validation_examples.jsonl")
    p.add_argument("--hypotheses", default="data/hypotheses.example.json")
    p.add_argument("--out", default="results/validation.md")
    p.add_argument("--out-json", default="results/validation.json")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--random-baselines", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dataset = load_jsonl(a.dataset)
    specs = json.loads(Path(a.hypotheses).read_text())
    from .common import load_model, down_projections
    model, tokenizer, device = load_model(a.model, a.device)

    texts = [r["text"] for r in dataset]
    scores = score_specs(model, tokenizer, device, texts, specs, a.max_tokens)
    neuron_count = down_projections(model)[0].in_features
    rng = random.Random(a.seed)

    results = []
    for spec_idx, spec in enumerate(specs):
        labels = [int(r[spec["label_field"]]) for r in dataset]
        spec_scores = [row[spec_idx] for row in scores]
        auc = auc_roc(labels, spec_scores)
        controls = []
        for control_neuron in pick_controls(neuron_count, {spec["neuron"]}, a.random_baselines, rng):
            control_spec = [{"layer": spec["layer"], "neuron": control_neuron, "polarity": spec.get("polarity", "positive")}]
            control_scores = [row[0] for row in score_specs(model, tokenizer, device, texts, control_spec, a.max_tokens)]
            c_auc = auc_roc(labels, control_scores)
            if c_auc is not None:
                controls.append(c_auc)
        control_mean = (sum(controls) / len(controls)) if controls else None
        delta = None if (auc is None or control_mean is None) else auc - control_mean
        results.append({
            "name": spec["name"],
            "layer": spec["layer"],
            "neuron": spec["neuron"],
            "polarity": spec.get("polarity", "positive"),
            "label_field": spec["label_field"],
            "positives": sum(labels),
            "auroc": auc,
            "control_mean_auroc": control_mean,
            "delta_vs_control": delta,
        })

    md = render_markdown(results, len(dataset), Path(a.dataset).name)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    Path(a.out_json).write_text(json.dumps({"results": results}, indent=2))
    print(f"Wrote {a.out} and {a.out_json}")


if __name__ == "__main__":
    main()
