from typing import List, Union
from .ATSMetrics.bernice_scorer import BerniceScorer


def corpus_bernice(orig_sents: Union[List[str], str], sys_sents: Union[List[str], str]):
    scorer = BerniceScorer()
    if isinstance(orig_sents, str):
        orig_sents = [orig_sents]
    if isinstance(sys_sents, str):
        sys_sents = [sys_sents]
    if len(orig_sents) < 2:
        print("BERNICE score can't be calculated with ")
    bernice_score = scorer.bernice_score(orig_sents, sys_sents)[0]
    return sum(bernice_score) / len(bernice_score)
