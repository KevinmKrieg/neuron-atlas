from src.visualize import render_svg


def test_visualize_mentions_neuron_and_layer():
    data = {
        "model": "Qwen/Qwen3-0.6B",
        "layers": [12],
        "top_k": 3,
        "num_lines": 4,
        "neurons": [
            {
                "layer": 12,
                "neuron": 947,
                "examples": [
                    {"activation": -5.25, "token": " player", "context": "The basketball player scored."},
                    {"activation": -1.95, "token": " player", "context": "The soccer player scored."},
                ],
            }
        ],
    }
    svg = render_svg(data, 12)
    assert "Neuron 947" in svg
    assert "Layer 12 summary" in svg
    assert "Qwen/Qwen3-0.6B" in svg
