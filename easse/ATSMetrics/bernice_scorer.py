from .toolsv2.bernice_tools import *
from .predictors.NSPredictor import NSPredictor

class BerniceScorer:
    def __init__(self):
        self.nspredictor = NSPredictor('bert-base-cased')

    # PURPOSE
    # Given a list of original sentences and a list of simplified sentences,
    # calculate and return the BERNICE score for document cohesion, a list
    # of tuples consisting of original incoherent pairs, their indices, and their confidencnes,
    # and a list of tuples consisting of simplified incoherent pairs, their indices, and their confidences.
    # The lists may be of different lengths and should be the
    # original and simplified version of the same document.
    # SIGNATURE
    # bernice_score :: List[String], List[String] => Float, List[Tuple], List[Tuple]
    def bernice_score(self, orig_sentences, simplified_sentences):
        orig_pairs = create_pairs(orig_sentences)
        orig_numeric = get_pairs_numeric(orig_sentences)
        num_orig_pairs = len(orig_pairs)
        simp_pairs = create_pairs(simplified_sentences)
        simp_numeric = get_pairs_numeric(simplified_sentences)
        # predictor = NSPredictor('bert-base-cased')
        orig_predictions = self.nspredictor.predict(orig_pairs)
        simp_predictions = self.nspredictor.predict(simp_pairs)
        orig_confidences = get_nsp_confidence(orig_pairs, self.nspredictor, orig_predictions)
        simp_confidences = get_nsp_confidence(simp_pairs, self.nspredictor, simp_predictions)
        orig_mean = get_avg_nsp_confidence(orig_confidences)
        simp_mean = get_avg_nsp_confidence(simp_confidences)
        orig_incoherent = count_invalid(orig_pairs, self.nspredictor, orig_predictions)
        simp_incoherent = count_invalid(simp_pairs, self.nspredictor, simp_predictions)
        score = calculate_bernice(simp_mean, orig_mean, simp_incoherent, orig_incoherent, num_orig_pairs)
        orig_low_pairs_info = self.get_low_scoring_pairs_info(orig_pairs, orig_numeric, orig_confidences)
        simp_low_pairs_info = self.get_low_scoring_pairs_info(simp_pairs, simp_numeric, simp_confidences)
        return score, orig_low_pairs_info, simp_low_pairs_info

    # PURPOSE
    # Given a list of pairs, a list of the pairs' indices in the sentence list,
    # and the pair confidences, return which pairs score low both as text and as indices.
    # SIGNATURE
    # get_low_scoring_pairs_info :: List[List], List[Tuple], List[Float] => List[Tuple]
    def get_low_scoring_pairs_info(self, pairs, pairs_numeric, confidences, low_score_threshold = 50):
        low_pairs_info = []
        for i, conf in enumerate(confidences):
            if conf < low_score_threshold:
                pair_indices = pairs_numeric[i]
                pair = pairs[i]
                low_pairs_info.append((pair_indices, pair, conf))
        return low_pairs_info
