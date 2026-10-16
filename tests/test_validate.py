from src.validate import auc_roc


def test_auc_roc_separates_classes():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert auc_roc(labels, scores) == 1.0
