import os
import argparse
import random
import jsonlines
import re
import ast
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
random.seed(42)
import argparse
import os
import sys
import pytrec_eval

#
# config
#
#parser = argparse.ArgumentParser()

#parser.add_argument('--train-dir', action='store', dest='train_dir',
#                    help='training file directory location', required=True)


#args = parser.parse_args()


def main(label_file, pred_file):
    #
    # load directory structure
    #

    labels = []

    with jsonlines.open(label_file, mode='r') as reader:
        for file in reader:
            labels.append(file)

    label_dict = {}
    for label in labels:
        label_dict.update({label.get('guid'): label.get('label')})

    print(len(labels))
    print(len(label_dict))

    with open(pred_file, 'r') as reader:
        content = reader.read().splitlines()
        predictions = [ast.literal_eval(file) for file in content]

    print(len(predictions))

    pred_dict = {}
    pos = []
    for pred in predictions:
        pred_dict.update({pred.get('guid'): pred.get('res')[1]})
        pos.append(pred.get('res')[1])

    print(min(pos))

    files = list(pred_dict.keys())
    files.sort()

    qrels = {}
    for file in files:
        qrels.update({file.split('_')[0]: {}})
    for file in files:
        print(file.split('_')[0])
        qrels.get(file.split('_')[0]).update({file.split('_')[1]: label_dict.get(file)})
        # qrels.update({file.split('_')[0]: {file.split('_')[1]: label_dict.get(file)}})
        # label_dict.get(file)

    run = {}
    for file in files:
        run.update({file.split('_')[0]: {}})
    for file in files:
        print(file.split('_')[0])
        run.get(file.split('_')[0]).update({file.split('_')[1]: pred_dict.get(file) + min(pos)})

    # trec eval

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    def write_line(measure, scope, value):
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            print_line(measure, query_id, value)

    # Scope hack: use query_measures of last item in previous loop to
    # figure out all unique measure names.
    #
    # TODO(cvangysel): add member to RelevanceEvaluator
    #                  with a list of measure names.
    for measure in sorted(query_measures.keys()):
        print_line(
            measure,
            'all',
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure]
                 for query_measures in results.values()]))

    with open(pred_file.split('.txt')[0] + '_eval.txt', 'w') as output:
        for measure in sorted(query_measures.keys()):
            output.write(write_line(
                measure,
                'all',
                pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure]
                     for query_measures in results.values()])) + '\n')




#label_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/val_org_top20.json'
#pred_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output_test_lawbert_lawattengru.txt'

label_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/test_org.json'
pred_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_patentbert_lawattengru.txt'

#main(label_file, pred_file)





# bm25 labels: from bm25_folder
#bm25_folder = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/bm25_top20_250'
bm25_folder = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/task1_test_bm25_top50'

#
# load directory structure
#

labels = []

with jsonlines.open(label_file, mode='r') as reader:
    for file in reader:
        labels.append(file)

label_dict = {}
for label in labels:
    label_dict.update({label.get('guid'): label.get('label')})

print(len(labels))
print(len(label_dict))

with open(pred_file, 'r') as reader:
    content = reader.read().splitlines()
    predictions = [ast.literal_eval(file) for file in content]

print(len(predictions))

pred_dict = {}
pos = []
for pred in predictions:
    pred_dict.update({pred.get('guid'): pred.get('res')[1]})
    pos.append(pred.get('res')[1])

print(min(pos))

files = list(pred_dict.keys())
files.sort()

qrels = {}
for file in files:
    qrels.update({file.split('_')[0]: {}})
for file in files:
    print(file.split('_')[0])
    qrels.get(file.split('_')[0]).update({file.split('_')[1]: label_dict.get(file)})
    # qrels.update({file.split('_')[0]: {file.split('_')[1]: label_dict.get(file)}})
    # label_dict.get(file)

run = {}
for file in files:
    run.update({file.split('_')[0]: {}})
for key in list(run.keys()):
    with open(os.path.join(bm25_folder, 'bm25_top50_{}.txt'.format(key)), 'r') as out:
        text = [text.split('_')[1] for text in
                [text.split('\n')[0].strip() for text in out.readlines()]]
        i = 0
        for doc in text:
            run.get(key).update({doc: len(text) - i})
            i = i + 1

# trec eval

evaluator = pytrec_eval.RelevanceEvaluator(
    qrels, pytrec_eval.supported_measures)

results = evaluator.evaluate(run)


def print_line(measure, scope, value):
    print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))


def write_line(measure, scope, value):
    return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)


for query_id, query_measures in sorted(results.items()):
    for measure, value in sorted(query_measures.items()):
        print_line(measure, query_id, value)

# Scope hack: use query_measures of last item in previous loop to
# figure out all unique measure names.
#
# TODO(cvangysel): add member to RelevanceEvaluator
#                  with a list of measure names.
for measure in sorted(query_measures.keys()):
    print_line(
        measure,
        'all',
        pytrec_eval.compute_aggregated_measure(
            measure,
            [query_measures[measure]
             for query_measures in results.values()]))

with open(os.path.join(bm25_folder, 'eval_bm25.txt'), 'w') as output:
    for measure in sorted(query_measures.keys()):
        output.write(write_line(
            measure,
            'all',
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure]
                 for query_measures in results.values()])) + '\n')



