# Neuron Atlas

**Mapping, interpreting, and causally validating native neurons in open-weight language models.**

Independent exploratory reproduction inspired by Nick Cammarata's [Neural Networks Are Already Clean](https://already-clean.goodfire.com/). Not affiliated with Goodfire. No empirical conclusions have been established.

## Research question
Do native MLP neurons in Qwen3-0.6B exhibit coherent activation patterns and causally meaningful effects without fitting a sparse autoencoder?

We hook the input to each selected MLP `down_proj`: `SiLU(gate_proj(x)) * up_proj(x)`. This is a **native MLP intermediate neuron**, not a residual-stream dimension.

## Quick start
Requires Python 3.10+ and a model download on first run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.scan --corpus data/prompts.txt --out results/scan.json --max-lines 30 --top-k 8
python -m src.report --scan results/scan.json --out results/report.md --layer 12
python -m src.intervene --layer 12 --neuron 100 --scale 0 --prompt 'The capital of France is'
pytest -q
```

By default the scan inspects layers 8, 12 and 16. `--device auto` selects CUDA, then MPS, then CPU. Pick actual candidate neuron IDs from the report before interventions.

## Workflow
1. **Scan**: gather strongest post-gating activations, token contexts, and signs.
2. **Explain**: hypothesize semantic functions for each native neuron/pole.
3. **Validate**: register predictions on unseen positive and negative examples; measure AUROC.
4. **Intervene**: suppress or scale a neuron and compare output with baseline.
5. **Compare**: eventually contrast native neurons and sparse autoencoder features.

## Scientific limitations
- Sample prompts are a smoke test, not representative of language.
- Highest activation does *not* imply semantic interpretability or causation.
- Greedy generation changes alone do not demonstrate a robust circuit.
- Goodfire's reported Qwen3-32B percentages are *not* independently reproduced here.
- Model-specific Hugging Face layer names may change.

## Roadmap
- [x] Native MLP activation hooks and top-context extraction
- [x] Per-layer Markdown reports
- [x] Neuron ablation/scaling demonstration
- [ ] Preregistered held-out neuron concept classification (AUROC)
- [ ] Random neuron and prompt controls
- [ ] Interactive neuron explorer
- [ ] SAE comparison and model-size study

## References
- [Goodfire essay](https://already-clean.goodfire.com/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

MIT License.
