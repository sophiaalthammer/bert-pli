import os
import argparse
import random
import pandas as pd
import jsonlines
import json
import re
import ast
import pytrec_eval
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
random.seed(42)

def read_label_file(label_file):
    with open(label_file, 'rb') as f:
        lines = json.load(f)

    labels = {}
    for key, value in lines.items():
        labels.update({key.strip('.txt'):{}})
        for doc in value:
            labels.get(key.strip('.txt')).update({doc.strip('.txt'):1})
    return labels


def read_predictions(pred_file):
    with open(pred_file, 'r') as f:
        lines = json.load(f)

    pred = {}
    for line in lines:
        query_id = line[0].split('_')[0]
        doc_id = line[0].split('_')[1]
        label = np.argmax(line[1])

        if pred.get(query_id):
            pred.get(query_id).update({doc_id: label})
        else:
            pred.update({query_id: {}})
            pred.get(query_id).update({doc_id: label})
    return pred


def ranking_eval(qrels, run, output_dir, output_file= 'eval_bm25_aggregate_overlap.txt'):
    # trec eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,  {'P_5', 'P_10', 'recall_5', 'recall_10', 'P_1'})
                                               #pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    def write_line(measure, scope, value):
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            print_line(measure, query_id, value)

    #for measure in sorted(query_measures.keys()):
    #    print_line(
    #        measure,
    #        'all',
    #        pytrec_eval.compute_aggregated_measure(
    #            measure,
    #            [query_measures[measure]
    #             for query_measures in results.values()]))

        with open(os.path.join(output_dir, output_file), 'w') as output:
            for measure in sorted(query_measures.keys()):
                output.write(write_line(
                    measure,
                    'all',
                    pytrec_eval.compute_aggregated_measure(
                        measure,
                        [query_measures[measure]
                         for query_measures in results.values()])) + '\n')




if __name__ == "__main__":
    #
    # config
    #
    #parser = argparse.ArgumentParser()

    #parser.add_argument('--label-file', action='store', dest='label_file',
    #                    help='json file with the labels of the test file', required=True)
    #parser.add_argument('--pred-file', action='store', dest='pred_file',
    #                    help='txt file with the binary predictions of the test file', required=False)
    #args = parser.parse_args()

    #label_file = args.label_file
    #pred_file = args.pred_file

    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json'
    pred_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/bertpli/output/' \
                'val_output/model_coliee22_attengru/output_val_merged.json'


    labels = read_label_file(label_file)
    predictions = read_predictions(pred_file)
    output_dir = '/'.join(pred_file.split('/')[:-1]) + 'eval.txt'

    ranking_eval(labels, predictions, output_dir, 'eval.txt')




