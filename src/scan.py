"""Record top absolute (signed) native MLP neuron activations over input tokens."""
import argparse
import heapq
import json
from pathlib import Path

import torch
from .common import load_model, down_projections, MODEL_ID


def scan(model, tokenizer, lines, device, top_k=8, layers=None, max_tokens=192):
    projections = down_projections(model)
    indices = [8, 12, 16] if layers is None else layers
    if any(i < 0 or i >= len(projections) for i in indices):
        raise ValueError("Layer index out of range")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    winners = {}
    handles = []
    active = {"line_id": None}

    def make_hook(layer_idx):
        def hook(module, inputs):
            tensor = inputs[0].detach().float()[0].cpu()
            for pos in range(tensor.shape[0]):
                for neuron, val in enumerate(tensor[pos].tolist()):
                    key = (layer_idx, neuron)
                    heap = winners.setdefault(key, [])
                    item = (abs(val), active["line_id"], pos, val)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
        return hook

    for i in indices:
        handles.append(projections[i].register_forward_pre_hook(make_hook(i)))
    texts = {}
    try:
        with torch.inference_mode():
            for line_id, line in enumerate(lines):
                texts[line_id] = line
                tokens = tokenizer(line, return_tensors="pt", truncation=True,
                                   max_length=max_tokens, add_special_tokens=False)
                if tokens["input_ids"].numel() == 0:
                    continue
                active["line_id"] = line_id
                model(**{k: v.to(device) for k, v in tokens.items()}, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    output = []
    for (layer, neuron), heap in sorted(winners.items()):
        examples = []
        for _, line_id, pos, value in sorted(heap, reverse=True):
            ids = tokenizer(texts[line_id], return_tensors="pt", truncation=True,
                            max_length=max_tokens, add_special_tokens=False)["input_ids"][0].tolist()
            examples.append({
                "line_id": line_id, "position": pos, "activation": round(value, 5),
                "token": tokenizer.decode(ids[pos:pos + 1]),
                "context": tokenizer.decode(ids[max(0, pos - 24):min(len(ids), pos + 8)]),
            })
        output.append({"layer": layer, "neuron": neuron, "examples": examples})
    return {"model": model.config._name_or_path, "layers": indices, "top_k": top_k,
            "num_lines": len(texts), "neurons": output}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--corpus", default="data/prompts.txt")
    p.add_argument("--out", default="results/scan.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-lines", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=192)
    args = p.parse_args()
    lines = [x.strip() for x in Path(args.corpus).read_text().splitlines()
             if x.strip()][:args.max_lines]
    model, tokenizer, device = load_model(args.model, args.device)
    output = scan(model, tokenizer, lines, device, args.top_k, args.layers, args.max_tokens)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {path} ({len(output['neurons'])} neuron records).")


if __name__ == "__main__":
    main()
