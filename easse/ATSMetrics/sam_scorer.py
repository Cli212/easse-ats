from .toolsv2.bernice_tools import *
from .predictors.NSPredictor import NSPredictor
from ctc_score import StyleTransferScorer, SummarizationScorer
import numpy as np
from tqdm import tqdm


def score_smoothing(x):
    return 1 + 4/(1+np.exp(-1.5*x-2))


class Scorer:
    def __init__(self, sam='align'):
        self.sam = sam
        if sam == 'align':
            self.alignment_predictor = StyleTransferScorer(align='E-bert', aggr_type='std')
        elif sam == 'rc':
            from .predictors.lercPredictor import lercPredictor
            from .predictors.QAPredictor import QAPredictor
            from .predictors.question_generation.pipelines import pipeline
            self.qa_generator = pipeline("question-generation", use_cuda=False)
            self.lerc_predictor = lercPredictor()
            self.qa_predictor = QAPredictor()
        else:
            raise NotImplementedError
        # self.nspredictor = NSPredictor('bert-base-cased')

    def sam_score(self, orig_sentences, simplified_sentences, question=None, answer=None):
        # orig_sentence, simplified_sentence = orig_sentences[0], simplified_sentences[0]
        scores = []

        for orig_sentence, simplified_sentence in tqdm(zip(orig_sentences, simplified_sentences), desc="Calculating SAM scores", total=len(orig_sentences)):
            if self.sam == 'align':
                sam_score = self.alignment_predictor.score(orig_sentence, simplified_sentence, aspect='preservation')[0]
            else:
                if not question or not answer:
                    try:
                        generated_questions = self.qa_generator(orig_sentence)
                    except Exception as e:
                        raise e
                    if len(generated_questions) == 0:
                        raise ValueError
                    question = generated_questions[0]['question']
                    answer = generated_questions[0]['answer']
                org_pred = self.qa_predictor.predict({"context": orig_sentence,
                                                 "question": question,
                                                 "reference": answer})[0]
                simp_pred = self.qa_predictor.predict({"context": simplified_sentence,
                                                 "question": question,
                                                 "reference": answer})[0]
                org_pred_score = self.lerc_predictor.predict({"candidate": org_pred,
                          "context": orig_sentence,
                          "question": question,
                          "reference": answer})
                simp_pred_score = self.lerc_predictor.predict({"candidate": simp_pred,
                                                              "context": orig_sentence,
                                                              "question": question,
                                                              "reference": answer})
                if org_pred_score > 5:
                    org_pred_score = 5
                elif org_pred_score < -5:
                    org_pred_score = -5
                if simp_pred_score > 5:
                    simp_pred_score = 5
                elif simp_pred_score < -5:
                    simp_pred_score = -5
                sam_score = score_smoothing(simp_pred_score - org_pred_score)
            scores.append(sam_score)
        return scores
    # PURPOSE
    # Given a list of original sentences and a list of simplified sentences,
    # calculate and return the BERNICE score for document cohesion, a list
    # of tuples consisting of original incoherent pairs, their indices, and their confidencnes,
    # and a list of tuples consisting of simplified incoherent pairs, their indices, and their confidences.
    # The lists may be of different lengths and should be the
    # original and simplified version of the same document.
    # SIGNATURE
    # bernice_score :: List[String], List[String] => Float, List[Tuple], List[Tuple]
    # def bernice_score(self, orig_sentences, simplified_sentences):
    #     orig_pairs = create_pairs(orig_sentences)
    #     orig_numeric = get_pairs_numeric(orig_sentences)
    #     num_orig_pairs = len(orig_pairs)
    #     simp_pairs = create_pairs(simplified_sentences)
    #     simp_numeric = get_pairs_numeric(simplified_sentences)
    #     # predictor = NSPredictor('bert-base-cased')
    #     orig_predictions = self.nspredictor.predict(orig_pairs)
    #     simp_predictions = self.nspredictor.predict(simp_pairs)
    #     orig_confidences = get_nsp_confidence(orig_pairs, self.nspredictor, orig_predictions)
    #     simp_confidences = get_nsp_confidence(simp_pairs, self.nspredictor, simp_predictions)
    #     orig_mean = get_avg_nsp_confidence(orig_confidences)
    #     simp_mean = get_avg_nsp_confidence(simp_confidences)
    #     orig_incoherent = count_invalid(orig_pairs, self.nspredictor, orig_predictions)
    #     simp_incoherent = count_invalid(simp_pairs, self.nspredictor, simp_predictions)
    #     score = calculate_bernice(simp_mean, orig_mean, simp_incoherent, orig_incoherent, num_orig_pairs)
    #     orig_low_pairs_info = self.get_low_scoring_pairs_info(orig_pairs, orig_numeric, orig_confidences)
    #     simp_low_pairs_info = self.get_low_scoring_pairs_info(simp_pairs, simp_numeric, simp_confidences)
    #     return score, orig_low_pairs_info, simp_low_pairs_info
    #
    # # PURPOSE
    # # Given a list of pairs, a list of the pairs' indices in the sentence list,
    # # and the pair confidences, return which pairs score low both as text and as indices.
    # # SIGNATURE
    # # get_low_scoring_pairs_info :: List[List], List[Tuple], List[Float] => List[Tuple]
    # def get_low_scoring_pairs_info(self, pairs, pairs_numeric, confidences, low_score_threshold = 50):
    #     low_pairs_info = []
    #     for i, conf in enumerate(confidences):
    #         if conf < low_score_threshold:
    #             pair_indices = pairs_numeric[i]
    #             pair = pairs[i]
    #             low_pairs_info.append((pair_indices, pair, conf))
    #     return low_pairs_info
