from rag.eval.metrics import accuracy, exact_match, f1_score


def test_exact_match_normalizes():
    assert exact_match("The Eiffel Tower.", ["eiffel tower"]) == 1.0


def test_f1_partial():
    assert 0 < f1_score("Paris France", ["Paris"]) < 1


def test_accuracy_substring():
    assert accuracy("It was built in Paris.", ["Paris"]) == 1.0
