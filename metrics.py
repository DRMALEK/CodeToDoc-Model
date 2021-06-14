# calculate rouge score, blue score and metor score for two files
import rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import FilesRouge
import argparse

class Metrics:
    
    @staticmethod
    def calculate_bleu(candidates_path, references_path):
        preds = open(candidates_path, 'r').readlines()
        refs = open(references_path, 'r').readlines()
        
        Ba = corpus_bleu(refs, preds)
        B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
        B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
        B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
        B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))

        Ba = round(Ba * 100, 2)
        B1 = round(B1 * 100, 2)
        B2 = round(B2 * 100, 2)
        B3 = round(B3 * 100, 2)
        B4 = round(B4 * 100, 2)

        ret = ''
        ret += ('for %s functions\n' % (len(preds)))
        ret += ('Ba %s\n' % (Ba))
        ret += ('B1 %s\n' % (B1))
        ret += ('B2 %s\n' % (B2))
        ret += ('B3 %s\n' % (B3))
        ret += ('B4 %s\n' % (B4))

        return ret
        
    @staticmethod
    def calculate_rouge(candidates_path, references_path):
        files_rouge = FilesRouge()
        scores = files_rouge.get_scores(references_path, candidates_path, avg=True)
        return scores

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0
        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1 