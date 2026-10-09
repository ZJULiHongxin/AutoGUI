"""Pure parsing of LLM <score> outputs for reject and verify stages."""
from __future__ import annotations


def parse_scores(resps: list, *, repeat: int, max_score: int) -> list:
    scores = []
    for resp in resps:
        score_start = resp.rfind("<score>") + 7
        equal_sign_id = resp.find("=", score_start)
        score_idx = equal_sign_id if equal_sign_id != -1 else score_start
        if score_idx == -1:
            continue
        while score_idx < len(resp) and not resp[score_idx].isnumeric():
            score_idx += 1
        try:
            score = min(repeat * max_score, int(resp[score_idx:score_idx + 2]))
            scores.append(score)
        except Exception:
            pass
    return scores


def parse_verification_scores(resps: list, *, verify_max_score: int):
    scores = [0] * (verify_max_score + 2)
    error_feedback = ""
    for resp in resps:
        score_start = resp.find("<score>") + 7
        score = resp[score_start:score_start + 1]
        if not score.isnumeric():
            error_feedback = (
                f'Invalid score: "{score}" detected! Assign a score ranging '
                f"from 0 to {verify_max_score}, enclosed within "
                f"<score></score> tags to reflect the degree to which the "
                f"candidate element meets the action's requirements. Now "
                f"please correct your output."
            )
            break
        scores[int(score)] += 1
    return scores, error_feedback
