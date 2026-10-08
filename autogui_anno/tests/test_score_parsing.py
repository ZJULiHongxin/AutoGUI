from autogui_anno.stages.scoring import parse_scores, parse_verification_scores

def test_parse_scores_wellformed():
    resps = ["blah <score>1", "stuff <score> = 0 </score>"]
    assert parse_scores(resps, repeat=1, max_score=1) == [1, 0]

def test_parse_scores_malformed_skipped():
    resps = ["no score tag here", "<score> not-a-number"]
    assert parse_scores(resps, repeat=1, max_score=1) == []

def test_parse_scores_empty_input():
    assert parse_scores([], repeat=1, max_score=1) == []

def test_parse_verification_scores_wellformed():
    resps = ["reasoning <score>3", "<score>2", "<score>3"]
    hist, err = parse_verification_scores(resps, verify_max_score=3)
    assert err == ""
    assert hist[3] == 2 and hist[2] == 1

def test_parse_verification_scores_malformed_returns_feedback():
    resps = ["<score>x"]
    hist, err = parse_verification_scores(resps, verify_max_score=3)
    assert err and "3" in err  # mentions verify_max_score, not NameError
