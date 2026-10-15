"""Small helpers for Hugging Face Qwen3 native MLP inspection."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def choose_device(requested="auto"):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id=MODEL_ID, device="auto"):
    device = choose_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)
    model.to(device).eval()
    return model, tokenizer, device


def down_projections(model):
    return [layer.mlp.down_proj for layer in model.model.layers]
