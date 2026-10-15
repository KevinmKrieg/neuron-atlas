from src.report import render


def test_report_contains_neurons():
    d = {"neurons": [{"layer": 2, "neuron": 42, "examples": [
        {"activation": -2.5, "token": " cat", "context": "the cat"}]}]}
    r = render(d, 2)
    assert "Neuron 42" in r
    assert "the cat" in r
    assert "negative poles" in r
