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
#parser = argparse.ArgumentParser()

#parser.add_argument('--train-dir', action='store', dest='train_dir',
#                    help='training file directory location', required=True)


#args = parser.parse_args()

label_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/test_org.json'
pred_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_patentbert_lawattengru2.txt'


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
