from ctc_score.scorer import Scorer


class StyleTransferScorer(Scorer):
    def __init__(self, align, aggr_type='mean', device='cuda'):
        Scorer.__init__(self, align=align, aggr_type=aggr_type, device=device)
        self.aligner = self._get_aligner('sent_to_sent')

    def score(self, input_sent, hypo, aspect, remove_stopwords=False, sim_func='cosine'):
        kwargs = dict(
            input_sent=input_sent,
            hypo=hypo,
            remove_stopwords=remove_stopwords,
            sim_func=sim_func)

        if aspect == 'preservation':
            return self.score_preservation(**kwargs)
        else:
            raise NotImplementedError

    def score_preservation(self, input_sent, hypo, remove_stopwords, sim_func):
        align_y_x = self.aligner.get_score(
            input_text=input_sent,
            context=hypo,
            remove_stopwords=remove_stopwords,
            sim_func=sim_func)

        align_x_y = self.aligner.get_score(
            input_text=hypo,
            context=input_sent,
            remove_stopwords=remove_stopwords,
            sim_func=sim_func)

        return 2 * (align_y_x * align_x_y) / (align_y_x + align_x_y), align_x_y, align_y_x

