"""Ablate/scale one native MLP neuron while generating comparable continuations."""
import argparse
import torch
from .common import load_model, down_projections, MODEL_ID


def generate_pair(model, tokenizer, device, prompt, layer, neuron, scale, max_new_tokens=50):
    projections = down_projections(model)
    if layer < 0 or layer >= len(projections):
        raise ValueError("Layer index out of range")
    if neuron < 0 or neuron >= projections[layer].in_features:
        raise ValueError("Neuron index out of range")
    batch = tokenizer(prompt, return_tensors="pt").to(device)

    def run():
        with torch.inference_mode():
            return tokenizer.decode(model.generate(
                **batch, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)

    baseline = run()

    def hook(module, inputs):
        x = inputs[0].clone()
        x[..., neuron] *= scale
        return (x,) + tuple(inputs[1:])

    handle = projections[layer].register_forward_pre_hook(hook)
    try:
        modified = run()
    finally:
        handle.remove()
    return baseline, modified


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="auto")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--neuron", type=int, required=True)
    p.add_argument("--scale", type=float, default=0.0)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=50)
    a = p.parse_args()
    model, tokenizer, device = load_model(a.model, a.device)
    baseline, modified = generate_pair(model, tokenizer, device, a.prompt,
                                       a.layer, a.neuron, a.scale, a.max_new_tokens)
    print("=== baseline ===\n" + baseline + "\n=== intervention ===\n" + modified)


if __name__ == "__main__":
    main()
