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


def read_predictions(pred_file, score=False):
    with open(pred_file, 'r') as f:
        lines = json.load(f)

    pred = {}
    for line in lines:
        query_id = line[0].split('_')[0]
        doc_id = line[0].split('_')[1]

        # take the argmax as label
        # take the score of the relevant dimension as label
        if not score:
            label = int(np.argmax(line[1]))
        else:
            label = int(line[1][1]*10000)

        if pred.get(query_id):
            pred.get(query_id).update({doc_id: label})
        else:
            pred.update({query_id: {}})
            pred.get(query_id).update({doc_id: label})
    return pred


def read_predictions_json(pred_file, score=False):
    with open(pred_file, 'r') as reader:
        content = reader.read().splitlines()
        predictions = [ast.literal_eval(file) for file in content]

    pred = {}
    for prediction in predictions:
        query_id = prediction.get('guid').split('_')[0]
        doc_id = prediction.get('guid').split('_')[1]
        if not score:
            label = int(np.argmax(prediction.get('res')))
        else:
            label = int(prediction.get('res')[1]*10000)

        if pred.get(query_id):
            pred.get(query_id).update({doc_id: label})
        else:
            pred.update({query_id: {}})
            pred.get(query_id).update({doc_id: label})
    return pred


def ranking_eval(qrels, run, output_dir, output_file= 'eval_bm25_aggregate_overlap.txt'):
    # trec eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, pytrec_eval.supported_measures)

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

def run_to_df(labels, columns):
    samples = []
    for query_id, value in labels.items():
        for doc_id, label in value.items():
            samples.append([query_id, doc_id, label])

    df_labels = pd.DataFrame(samples)
    df_labels = df_labels.rename(columns={0: columns[0], 1: columns[1], 2: columns[2]})
    return df_labels


def eval_classification(labels, predictions, output_dir, output_file):
    df_labels = run_to_df(labels, ['query_id', 'doc_id', 'label'])
    df_pred = run_to_df(predictions, ['query_id', 'doc_id', 'prediction'])

    assert set(df_labels['query_id']) == set(df_pred['query_id'])

    # join outer
    df_joined = pd.merge(df_labels, df_pred, how='outer', on=['query_id', 'doc_id'])
    df_joined = df_joined.fillna(0)

    # manual check if the precision, recall and f1 score of the sklearn binary classification package (for relevant class 1)
    #     return the same evaluation metrics as when they are computed manually according to the formula on
    #     https://sites.ualberta.ca/~rabelo/COLIEE2020/

    label_list = list(df_joined['label'])
    pred_list = list(df_joined['prediction'])
    df = df_joined

    df_only_rel = df[df['label'] == 1]
    try:
        no_correctly_retrieved = df_only_rel['prediction'].value_counts()[1]
    except:
        no_correctly_retrieved = 0

    no_relevant = len(df_only_rel)
    try:
        df_only_rel_pred = df[df['prediction'] == 1]
        no_retrieved = len(df_only_rel_pred)
    except:
        no_retrieved = 0

    if no_retrieved != 0:
        precision = no_correctly_retrieved / no_retrieved
    else:
        precision = 0
    recall = no_correctly_retrieved / no_relevant
    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    with open(os.path.join(output_dir,output_file), 'w+') as output:
        output.write(classification_report(label_list, pred_list) + '\n')
        output.write("Precision (for [0,1] class): {} \n".format(precision_score(label_list, pred_list, labels=[0, 1], average = None,
                                                                  pos_label=1)))
        output.write("Recall (for [0,1] class): {}\n".format(recall_score(label_list, pred_list, average= None)))
        output.write("F1 score (for [0,1] class): {} \n".format(f1_score(label_list, pred_list, average=None), end='\n\n'))

        output.write("Manually calculated Precision: {} \n".format(precision))
        output.write("Manually calculated Recall: {}\n".format(recall))
        output.write("Manually calculated F1 score: {} \n".format(f1, end='\n\n'))


def format_pred_ranking_to_binary(predictions, cutoff=6):
    pred_cutoff = {}
    for query_id, ranking in predictions.items():
        tuples = [(doc_id, score) for doc_id, score in ranking.items()]
        tuples.sort(key=lambda x: x[1], reverse=True)
        tuples_cutoff = [(tuples[i][0], 1) if i< cutoff else (tuples[i][0],0) for i in range(len(tuples))]
        sorted_dict = {k: v for k, v in tuples_cutoff}
        pred_cutoff.update({query_id:sorted_dict})
    return pred_cutoff


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
                'val_output/model_coliee22_attengru_pluspos/output_val_merged_balanced.json'

    labels = read_label_file(label_file)
    if 'lstm' in pred_file:
        predictions = read_predictions_json(pred_file, score=False)
    else:
        predictions = read_predictions(pred_file, score=False)
    output_dir = '/'.join(pred_file.split('/')[:-1])

    ranking_eval(labels, predictions, output_dir, 'eval_balanced.txt')
    eval_classification(labels, predictions, output_dir, 'eval_classification_balanced.txt')

    # eval classification with scores! cutoff!
    # this is better!
    pred_cutoff = format_pred_ranking_to_binary(predictions, cutoff=6)   #eval different cutoffs, eval different reranking depths later!
    eval_classification(labels, pred_cutoff, output_dir, 'eval_balanced_scores_cutoff6.txt')



