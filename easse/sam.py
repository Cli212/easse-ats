from typing import List, Union
from .ATSMetrics.sam_scorer import Scorer


def corpus_sam(orig_sents: Union[List[str], str], sys_sents: Union[List[str], str], sam='align'):
    scorer = Scorer(sam=sam)
    if isinstance(orig_sents, str):
        orig_sents = [orig_sents]
    if isinstance(sys_sents, str):
        sys_sents = [sys_sents]
    sam_score = scorer.sam_score(orig_sents, sys_sents)
    return sum(sam_score) / len(sam_score), sam_score

