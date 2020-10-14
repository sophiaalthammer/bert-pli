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

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--label-file', action='store', dest='label_file',
                    help='json file with the labels of the test file', required=True)
parser.add_argument('--pred-file', action='store', dest='pred_file',
                    help='txt file with the binary predictions of the test file', required=True)

args = parser.parse_args()

#label_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/test_org.json'
#pred_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_lawattengru.txt'

#label_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/val_org_top20.json'
#pred_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output_test_patentbert_lawattenlstm2.txt'


#
# load directory structure
#

labels = []

with jsonlines.open(args.label_file, mode='r') as reader:
    for file in reader:
        labels.append(file)

label_dict = {}
for label in labels:
    label_dict.update({label.get('guid'): label.get('label')})

print(len(labels))
print(len(label_dict))

with open(args.pred_file, 'r') as reader:
    content = reader.read().splitlines()
    predictions = [ast.literal_eval(file) for file in content]

print(len(predictions))

pred_dict = {}
for pred in predictions:
    pred_dict.update({pred.get('guid'): np.argmax(pred.get('res'))})


files = list(pred_dict.keys())
files.sort()

label_list = []
pred_list = []
for file in files:
    label_list.append(label_dict.get(file))
    pred_list.append(pred_dict.get(file))

assert len(label_list) == len(pred_list)

print(classification_report(label_list, pred_list))

# These values are the same as in the table above

print("Precision (macro): %f" % precision_score(label_list, pred_list, labels=[0,1], average='macro', pos_label=1))
print("Recall (macro):    %f" % recall_score(label_list, pred_list, average='macro'))
print("F1 score (macro):  %f" % f1_score(label_list, pred_list, average='macro'), end='\n\n')
